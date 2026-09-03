from std.memory import alloc
from std.math import tan, atan2, sqrt, cos, sin
from std.ffi import external_call
from .lexer import is_whitespace
from .parse_types import SceneParseState, NamedMaterial
from .geometry import RGB, MatKind, PI, Vec3f
from .transform import matrix_invert, transform_normals
from .scene_builder import store_mesh
from .mitsuba_serialized import load_mitsuba_serialized, MitsubaMesh
from .pbrt_parser import ParsedScene_Mojo, finalize_scene

# ── Mitsuba 0.5/0.6/2/3 XML scene parser ────────────────────────────────────
#
# Originally scoped to the old (camelCase, matrix-only-transform) dialect
# needed for the torus-in-glass caustic scene -- see project_mitsuba_parser
# memory. Extended 2026-08-07 to also handle the modern (snake_case,
# composable-transform) Mitsuba 2/3 dialect, needed to render the original
# SMS paper's own example scenes (~/src/specular-manifold-sampling). Both
# dialects are supported simultaneously via dual attribute-name lookups
# (`_mxml_find_attr2`) and a unified transform-block parser
# (`_mit_parse_transform_block`) that handles a bare `<matrix>` (old
# dialect) exactly like before, or composes `<translate>`/`<rotate>`/
# `<scale>`/`<lookat>` children (modern dialect) into the same matrix shape.
#
# In scope: <sensor type="perspective"> (camera/film/sampler, crop window),
# <bsdf type="diffuse"|"dielectric"|"thindielectric"|"roughdielectric">
# (top-level named, inline unnamed, or wrapped in normalmap/bumpmap/
# twosided -- unwrapped to the inner bsdf), <shape type="serialized"|
# "rectangle"> with an optional nested <emitter type="area">,
# `<default name=.../>` + `$name` substitution. `<shape type="sphere">` is
# explicitly NOT supported -- see the long comment at its dispatch site.
# Anything else (cube/ply/instance/envmap shapes, other bsdf types,
# non-area emitters, textures) is explicitly out of scope and warned about,
# not silently ignored.
#
# Populates the SAME `SceneParseState` -> `finalize_scene` -> `ParsedScene_Mojo`
# pipeline the pbrt parser uses -- see scene_builder.mojo's module docstring
# for why that back end needs no changes at all for a new front-end.

# ── Hand-rolled XML tokenizer ────────────────────────────────────────────────

@fieldwise_init
struct MitsubaAttr(Copyable, ImplicitlyCopyable, Movable):
    var name:  String
    var value: String

struct MitsubaTag(Copyable, Movable):
    var name:          String
    var attrs:         List[MitsubaAttr]
    var is_close:      Bool
    var is_self_close: Bool

    def __init__(out self, name: String, is_close: Bool, is_self_close: Bool):
        self.name = name
        self.attrs = List[MitsubaAttr]()
        self.is_close = is_close
        self.is_self_close = is_self_close

def _mxml_make_string(buf: UnsafePointer[UInt8, MutExternalOrigin], start: Int, end: Int) -> String:
    var n = end - start
    var tmp = alloc[UInt8](n + 1)
    for i in range(n):
        tmp[i] = buf[start + i]
    tmp[n] = UInt8(0)
    var s = String(unsafe_from_utf8_ptr=tmp.as_immutable())
    tmp.free()
    return s

def _mxml_find_attr(tag: MitsubaTag, name: String) -> String:
    for i in range(len(tag.attrs)):
        if tag.attrs[i].name == name:
            return tag.attrs[i].value
    return String("")

def _mxml_skip_block(tags: List[MitsubaTag], start: Int) -> Int:
    """`start` indexes a non-self-closing open tag. Returns the index one
    past its matching close tag -- name-agnostic depth counting, safe since
    well-formed XML always closes in LIFO order."""
    var depth = 1
    var i = start + 1
    var n = len(tags)
    while i < n and depth > 0:
        if tags[i].is_close:
            depth -= 1
        elif not tags[i].is_self_close:
            depth += 1
        i += 1
    return i

def _mit_block_end(tags: List[MitsubaTag], idx: Int) -> Int:
    if tags[idx].is_self_close:
        return idx + 1
    return _mxml_skip_block(tags, idx)

def _mit_find_child(tags: List[MitsubaTag], start: Int, end: Int, tagname: String) -> Int:
    for j in range(start, end):
        if not tags[j].is_close and tags[j].name == tagname:
            return j
    return -1

def _mit_find_child_by_attr(tags: List[MitsubaTag], start: Int, end: Int,
                            tagname: String, attr_name: String, attr_val: String) -> Int:
    for j in range(start, end):
        if not tags[j].is_close and tags[j].name == tagname:
            if _mxml_find_attr(tags[j], attr_name) == attr_val:
                return j
    return -1

def _mit_find_transform(tags: List[MitsubaTag], start: Int, end: Int) -> Int:
    """A `<transform name="toWorld">` (old dialect) or
    `<transform name="to_world">` (modern dialect) child, whichever is
    present."""
    var idx = _mit_find_child_by_attr(tags, start, end, "transform", "name", "toWorld")
    if idx >= 0:
        return idx
    return _mit_find_child_by_attr(tags, start, end, "transform", "name", "to_world")

def _mit_find_value_tag(tags: List[MitsubaTag], start: Int, end: Int, param_name: String) -> Int:
    """A `<rgb name="param_name" .../>` or `<spectrum name="param_name" .../>`
    child, whichever is present -- modern Mitsuba scenes often write scalar
    OR RGB-valued params as `<spectrum>` (e.g. `<spectrum name="radiance"
    value="50000"/>`), while the old dialect and some modern scenes still
    use `<rgb>` for genuinely 3-channel values."""
    var idx = _mit_find_child_by_attr(tags, start, end, "rgb", "name", param_name)
    if idx >= 0:
        return idx
    return _mit_find_child_by_attr(tags, start, end, "spectrum", "name", param_name)

def tokenize_mitsuba_xml(buf: UnsafePointer[UInt8, MutExternalOrigin], length: Int) -> List[MitsubaTag]:
    var tags = List[MitsubaTag]()
    var pos = 0
    while pos < length:
        while pos < length and buf[pos] != UInt8(60):  # '<'
            pos += 1
        if pos >= length:
            break
        if pos + 1 < length and buf[pos + 1] == UInt8(63):  # '<?' ... '?>'
            pos += 2
            while pos + 1 < length and not (buf[pos] == UInt8(63) and buf[pos + 1] == UInt8(62)):
                pos += 1
            pos += 2
            continue
        if pos + 3 < length and buf[pos + 1] == UInt8(33) and buf[pos + 2] == UInt8(45) and buf[pos + 3] == UInt8(45):
            # '<!--' ... '-->'
            pos += 4
            while pos + 2 < length and not (buf[pos] == UInt8(45) and buf[pos + 1] == UInt8(45) and buf[pos + 2] == UInt8(62)):
                pos += 1
            pos += 3
            continue
        if pos + 1 < length and buf[pos + 1] == UInt8(33):  # other '<! ... >'
            pos += 2
            while pos < length and buf[pos] != UInt8(62):
                pos += 1
            pos += 1
            continue

        var is_close = False
        pos += 1  # consume '<'
        if pos < length and buf[pos] == UInt8(47):  # '/'
            is_close = True
            pos += 1
        var name_start = pos
        while pos < length and not is_whitespace(buf[pos]) and buf[pos] != UInt8(62) and buf[pos] != UInt8(47):
            pos += 1
        var tag = MitsubaTag(_mxml_make_string(buf, name_start, pos), is_close, False)

        while True:
            while pos < length and is_whitespace(buf[pos]):
                pos += 1
            if pos >= length:
                break
            if buf[pos] == UInt8(47):  # '/>'
                tag.is_self_close = True
                pos += 1
                while pos < length and buf[pos] != UInt8(62):
                    pos += 1
                pos += 1
                break
            if buf[pos] == UInt8(62):  # '>'
                pos += 1
                break
            var aname_start = pos
            while pos < length and buf[pos] != UInt8(61) and not is_whitespace(buf[pos]) and buf[pos] != UInt8(62) and buf[pos] != UInt8(47):
                pos += 1
            var aname = _mxml_make_string(buf, aname_start, pos)
            while pos < length and is_whitespace(buf[pos]):
                pos += 1
            var aval = String("")
            if pos < length and buf[pos] == UInt8(61):  # '='
                pos += 1
                while pos < length and is_whitespace(buf[pos]):
                    pos += 1
                if pos < length and (buf[pos] == UInt8(34) or buf[pos] == UInt8(39)):  # quote
                    var q = buf[pos]
                    pos += 1
                    var vstart = pos
                    while pos < length and buf[pos] != q:
                        pos += 1
                    aval = _mxml_make_string(buf, vstart, pos)
                    if pos < length:
                        pos += 1  # consume closing quote
            tag.attrs.append(MitsubaAttr(aname, aval))
        tags.append(tag^)
    return tags^

# ── Minimal numeric parsing (String -> Float32) ─────────────────────────────
# Mitsuba attribute values are plain ASCII decimal literals (with an optional
# sign/exponent) -- this is a small standalone reimplementation of the same
# grammar lexer.mojo's scan_float already handles for pbrt tokens, kept
# separate since it operates on a String's bytes rather than a scanner
# handle's cursor.

@always_inline
def _mit_is_sep(b: UInt8) -> Bool:
    # Space/tab/newline/cr, plus comma -- modern Mitsuba XML writes vector
    # attributes like `origin="10, 11, 10"` comma-separated, unlike the old
    # dialect's plain-space-separated <matrix value="...">.
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13) or b == UInt8(44)

def _mit_parse_float_at(s: String, start: Int) -> Tuple[Float32, Int]:
    var bytes = s.as_bytes()
    var n = len(bytes)
    var i = start
    while i < n and _mit_is_sep(bytes[i]):
        i += 1
    var neg = False
    if i < n and bytes[i] == UInt8(45):
        neg = True
        i += 1
    var ip = Float64(0)
    while i < n and bytes[i] >= UInt8(48) and bytes[i] <= UInt8(57):
        ip = ip * Float64(10) + Float64(Int(bytes[i]) - 48)
        i += 1
    if i < n and bytes[i] == UInt8(46):
        i += 1
        var frac = Float64(0.1)
        while i < n and bytes[i] >= UInt8(48) and bytes[i] <= UInt8(57):
            ip += Float64(Int(bytes[i]) - 48) * frac
            frac *= Float64(0.1)
            i += 1
    if i < n and (bytes[i] == UInt8(101) or bytes[i] == UInt8(69)):  # e/E
        i += 1
        var eneg = False
        if i < n and bytes[i] == UInt8(45):
            eneg = True
            i += 1
        elif i < n and bytes[i] == UInt8(43):
            i += 1
        var ev = 0
        while i < n and bytes[i] >= UInt8(48) and bytes[i] <= UInt8(57):
            ev = ev * 10 + Int(bytes[i]) - 48
            i += 1
        var factor = Float64(1)
        for _ in range(ev):
            factor *= Float64(10)
        if eneg:
            ip /= factor
        else:
            ip *= factor
    if neg:
        ip = -ip
    return (Float32(ip), i)

def _mit_parse_float(s: String) -> Float32:
    var (v, _) = _mit_parse_float_at(s, 0)
    return v

def _mit_parse_floats(s: String) -> List[Float32]:
    var result = List[Float32]()
    var bytes = s.as_bytes()
    var n = len(bytes)
    var i = 0
    while i < n:
        while i < n and _mit_is_sep(bytes[i]):
            i += 1
        if i >= n:
            break
        var (v, ni) = _mit_parse_float_at(s, i)
        result.append(v)
        if ni <= i:
            break
        i = ni
    return result^

def _mit_string_from(s: String, start: Int) -> String:
    var bytes = s.as_bytes()
    var n = len(bytes)
    var buf = alloc[UInt8](n - start + 1)
    for i in range(start, n):
        buf[i - start] = bytes[i]
    buf[n - start] = UInt8(0)
    var r = String(unsafe_from_utf8_ptr=buf.as_immutable())
    buf.free()
    return r

def _mit_apply_defaults(mut tags: List[MitsubaTag]):
    """Modern Mitsuba scenes parameterize values via
    `<default name="X" value="Y"/>` + `$X` references elsewhere (e.g.
    `<integer name="sample_count" value="$spp"/>`). gonzales doesn't
    implement Mitsuba's full CLI `-Dname=value` override system -- just
    enough substitution to make a scene's OWN declared defaults resolve:
    every `$name` attribute value is replaced by the matching <default>'s
    own value string, in document order (a later <default> for the same
    name overrides an earlier one, matching Mitsuba's own last-wins
    semantics)."""
    var names = List[String]()
    var values = List[String]()
    for i in range(len(tags)):
        if not tags[i].is_close and tags[i].name == "default":
            var dname = _mxml_find_attr(tags[i], "name")
            var dval = _mxml_find_attr(tags[i], "value")
            if dname != String(""):
                names.append(dname)
                values.append(dval)
    for i in range(len(tags)):
        ref t = tags[i]
        for a in range(len(t.attrs)):
            ref at = t.attrs[a]
            var vb = at.value.as_bytes()
            if len(vb) > 1 and vb[0] == UInt8(36):  # '$'
                var refname = _mit_string_from(at.value, 1)
                for k in range(len(names)):
                    if names[k] == refname:
                        at.value = values[k]
                        break

# ── Matrix helpers ───────────────────────────────────────────────────────────

def _mit_identity_ctm() -> InlineArray[Float32, 16]:
    var m = InlineArray[Float32, 16](fill=Float32(0))
    m[0] = Float32(1); m[5] = Float32(1); m[10] = Float32(1); m[15] = Float32(1)
    return m^

def _mit_matrix_rowmajor_to_ctm(vals: List[Float32]) -> InlineArray[Float32, 16]:
    """Mitsuba XML <matrix value="..."> is row-major 16 floats
    (flat[row*4+col]); gonzales's CTM is column-major (flat[col*4+row]) --
    see transform.mojo:4-5. A plain transpose is enough for a shape's
    toWorld (== gonzales's ordinary object-to-world CTM, no inversion
    needed, same semantics as pbrt's CTM for Shape directives)."""
    if len(vals) < 16:
        return _mit_identity_ctm()
    var m = InlineArray[Float32, 16](fill=Float32(0))
    for r in range(4):
        for c in range(4):
            m[c * 4 + r] = vals[r * 4 + c]
    return m^

def dot3(a: Vec3f, b: Vec3f) -> Float32:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross3(a: Vec3f, b: Vec3f) -> Vec3f:
    return Vec3f(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _mit_parse_vec3_attr(tag: MitsubaTag, name: String) -> Vec3f:
    """A vector given either as three separate x/y/z attributes
    (`<translate x="1" y="2" z="3"/>`) or one comma-separated attribute
    (`<lookat origin="1, 2, 3" .../>`)."""
    var combo = _mxml_find_attr(tag, name)
    if combo != String(""):
        var vals = _mit_parse_floats(combo)
        if len(vals) >= 3:
            return Vec3f(vals[0], vals[1], vals[2])
    var xs = _mxml_find_attr(tag, "x")
    var ys = _mxml_find_attr(tag, "y")
    var zs = _mxml_find_attr(tag, "z")
    var x = _mit_parse_float(xs) if xs != String("") else Float32(0)
    var y = _mit_parse_float(ys) if ys != String("") else Float32(0)
    var z = _mit_parse_float(zs) if zs != String("") else Float32(0)
    return Vec3f(x, y, z)

def _mit_rm_identity() -> List[Float32]:
    var m = List[Float32]()
    m.resize(16, Float32(0))
    m[0] = Float32(1); m[5] = Float32(1); m[10] = Float32(1); m[15] = Float32(1)
    return m^

def _mit_rm_mul(a: List[Float32], b: List[Float32]) -> List[Float32]:
    """Row-major 4x4 multiply: result = a * b (a applied AFTER b to a point,
    i.e. result·p = a·(b·p)) -- matches modern Mitsuba's own <transform>
    composition, where each subsequent child element is applied after the
    ones already accumulated (scale-then-rotate-then-translate reads
    top-to-bottom as "first scale, then rotate the scaled thing, ...")."""
    var r = List[Float32]()
    r.resize(16, Float32(0))
    for row in range(4):
        for col in range(4):
            var s = Float32(0)
            for k in range(4):
                s += a[row * 4 + k] * b[k * 4 + col]
            r[row * 4 + col] = s
    return r^

def _mit_rm_translate(x: Float32, y: Float32, z: Float32) -> List[Float32]:
    var m = _mit_rm_identity()
    m[3] = x; m[7] = y; m[11] = z
    return m^

def _mit_rm_scale(x: Float32, y: Float32, z: Float32) -> List[Float32]:
    var m = List[Float32]()
    m.resize(16, Float32(0))
    m[0] = x; m[5] = y; m[10] = z; m[15] = Float32(1)
    return m^

def _mit_rm_rotate(axis: Vec3f, angle_deg: Float32) -> List[Float32]:
    var alen = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    var m = _mit_rm_identity()
    if alen < Float32(1e-12):
        return m^
    var x = axis[0]/alen; var y = axis[1]/alen; var z = axis[2]/alen
    var rad = angle_deg * PI / Float32(180)
    var c = cos(rad); var s = sin(rad)
    var t = Float32(1) - c
    m[0]  = t*x*x + c;    m[1]  = t*x*y - s*z;  m[2]  = t*x*z + s*y
    m[4]  = t*x*y + s*z;  m[5]  = t*y*y + c;    m[6]  = t*y*z - s*x
    m[8]  = t*x*z - s*y;  m[9]  = t*y*z + s*x;  m[10] = t*z*z + c
    return m^

def _mit_rm_lookat(origin: Vec3f, target: Vec3f, up: Vec3f) -> List[Float32]:
    """Mitsuba's own camera convention: forward = +Z, right-handed
    (X x Y = Z) -- confirmed empirically against ori_torus.xml's own sensor
    matrix (see project_mitsuba_parser memory). Produces a camera-to-world
    matrix in the SAME row-major shape a `<matrix value="...">` would."""
    var dir = target - origin
    var dl = sqrt(dot3(dir, dir))
    if dl > Float32(1e-12):
        dir = dir * (Float32(1)/dl)
    var upn = up
    var ul = sqrt(dot3(upn, upn))
    if ul > Float32(1e-12):
        upn = upn * (Float32(1)/ul)
    var right = cross3(upn, dir)
    var rl = sqrt(dot3(right, right))
    if rl > Float32(1e-12):
        right = right * (Float32(1)/rl)
    var newUp = cross3(dir, right)
    var m = _mit_rm_identity()
    m[0] = right[0]; m[1] = newUp[0]; m[2] = dir[0]; m[3] = origin[0]
    m[4] = right[1]; m[5] = newUp[1]; m[6] = dir[1]; m[7] = origin[1]
    m[8] = right[2]; m[9] = newUp[2]; m[10] = dir[2]; m[11] = origin[2]
    return m^

def _mit_parse_transform_block(tags: List[MitsubaTag], tf_idx: Int, tf_end: Int) -> List[Float32]:
    """Parses every child of a `<transform>` element in document order and
    composes them into one row-major 16-float matrix, matching Mitsuba's
    own semantics: each subsequent child is applied AFTER everything
    accumulated so far (`accumulated = child_matrix * accumulated`).
    Handles both the old dialect's single `<matrix value="...">` (a
    one-child "sequence" of length 1, unchanged behavior from before this
    function existed) and the modern dialect's composable
    `<translate>`/`<rotate>`/`<scale>`/`<lookat>` children."""
    var acc = _mit_rm_identity()
    var j = tf_idx + 1
    while j < tf_end - 1:
        ref t = tags[j]
        if t.is_close:
            j += 1
            continue
        var child_end = _mit_block_end(tags, j)
        if t.name == "matrix":
            var vals = _mit_parse_floats(_mxml_find_attr(t, "value"))
            if len(vals) >= 16:
                var cm = List[Float32]()
                for k in range(16):
                    cm.append(vals[k])
                acc = _mit_rm_mul(cm, acc)
        elif t.name == "translate":
            var v = _mit_parse_vec3_attr(t, "value")
            acc = _mit_rm_mul(_mit_rm_translate(v[0], v[1], v[2]), acc)
        elif t.name == "scale":
            var sval = _mxml_find_attr(t, "value")
            var v: Vec3f
            if sval != String(""):
                var sv = _mit_parse_float(sval)
                v = Vec3f(sv, sv, sv)
            else:
                v = _mit_parse_vec3_attr(t, "value")
                if v[0] == Float32(0) and v[1] == Float32(0) and v[2] == Float32(0):
                    v = Vec3f(Float32(1), Float32(1), Float32(1))
            acc = _mit_rm_mul(_mit_rm_scale(v[0], v[1], v[2]), acc)
        elif t.name == "rotate":
            var axis = _mit_parse_vec3_attr(t, "axis")
            var angle_str = _mxml_find_attr(t, "angle")
            var angle = _mit_parse_float(angle_str) if angle_str != String("") else Float32(0)
            acc = _mit_rm_mul(_mit_rm_rotate(axis, angle), acc)
        elif t.name == "lookat":
            var origin = _mit_parse_vec3_attr(t, "origin")
            var target = _mit_parse_vec3_attr(t, "target")
            var up = _mit_parse_vec3_attr(t, "up")
            acc = _mit_rm_mul(_mit_rm_lookat(origin, target, up), acc)
        j = child_end
    return acc^

def _mit_shorter_axis_fov(fov_deg: Float32, fov_axis: String, film_w: Int32, film_h: Int32) -> Float32:
    """gonzales/pbrt's `camera_fov` is the FOV of the image's narrower axis
    (see make_perspective_matrix/finalize_scene's screen-window setup);
    Mitsuba specifies fov along a named axis ("x"/"y", or modern dialect's
    "smaller"/"larger", meaning exactly what it says -- "smaller" already
    IS the narrower-axis fov gonzales wants directly, no conversion).
    Convert via the tan(half-fov) ratio, which is exactly `width/height`
    between the x/y axes for a pinhole camera."""
    if fov_axis == "smaller":
        return fov_deg
    var aspect = Float32(film_w) / Float32(film_h)
    if fov_axis == "larger":
        var half_rad_l = fov_deg * PI / Float32(360)
        var tan_half_l = tan(half_rad_l)
        var narrow_l = tan_half_l / aspect if film_w >= film_h else tan_half_l * aspect
        return atan2(narrow_l, Float32(1)) * Float32(360) / PI
    var half_rad = fov_deg * PI / Float32(360)
    var tan_half = tan(half_rad)
    var tan_half_x: Float32
    var tan_half_y: Float32
    if fov_axis == "y":
        tan_half_y = tan_half
        tan_half_x = tan_half * aspect
    else:
        tan_half_x = tan_half
        tan_half_y = tan_half / aspect
    var narrow_tan_half = tan_half_x if film_w <= film_h else tan_half_y
    return atan2(narrow_tan_half, Float32(1)) * Float32(360) / PI

# ── Sensor (camera/film/sampler) ─────────────────────────────────────────────

def _mit_process_sensor(tags: List[MitsubaTag], start: Int, end: Int,
                        s_ptr: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var tf_idx = _mit_find_transform(tags, start, end)
    if tf_idx >= 0:
        var tf_end = _mit_block_end(tags, tf_idx)
        var floats = _mit_parse_transform_block(tags, tf_idx, tf_end)
        # Mitsuba's toWorld is already camera-to-world (unlike pbrt,
        # where the CTM-at-Camera-time is world-to-camera) -- so
        # after the row-major -> column-major transpose, invert it
        # once here so finalize_scene's own re-inversion of
        # cam2w_raw recovers the original toWorld. See
        # project_mitsuba_parser memory for the full derivation.
        var c2w_col = _mit_matrix_rowmajor_to_ctm(floats)
        # Mitsuba's camera-space X axis points the opposite way from
        # gonzales/pbrt's convention (confirmed empirically: without
        # this negation, the render comes out exactly left-right
        # mirrored vs. Mitsuba's own reference render of this scene,
        # e.g. the floor caustic streak lands on the wrong side).
        # Y (up) and Z (forward) both check out unnegated already --
        # only X needs flipping.
        c2w_col[0] = -c2w_col[0]
        c2w_col[1] = -c2w_col[1]
        c2w_col[2] = -c2w_col[2]
        var c2w_arr = alloc[Float32](16)
        for k in range(16):
            c2w_arr[k] = c2w_col[k]
        var w2c = alloc[Float32](16)
        _ = matrix_invert(c2w_arr, w2c)
        for k in range(16):
            s_ptr[0].cam2w_raw[k] = w2c[k]
        c2w_arr.free(); w2c.free()

    var fov_val = Float32(30)
    var fov_idx = _mit_find_child_by_attr(tags, start, end, "float", "name", "fov")
    if fov_idx >= 0:
        fov_val = _mit_parse_float(_mxml_find_attr(tags[fov_idx], "value"))
    var fov_axis = String("x")
    var axis_idx = _mit_find_child_by_attr(tags, start, end, "string", "name", "fovAxis")
    if axis_idx < 0:
        axis_idx = _mit_find_child_by_attr(tags, start, end, "string", "name", "fov_axis")
    if axis_idx >= 0:
        fov_axis = _mxml_find_attr(tags[axis_idx], "value")

    var film_w = s_ptr[0].film_w
    var film_h = s_ptr[0].film_h
    var film_idx = _mit_find_child(tags, start, end, "film")
    if film_idx >= 0:
        var film_end = _mit_block_end(tags, film_idx)
        var w_idx = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "width")
        var h_idx = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "height")
        if w_idx >= 0:
            film_w = Int32(_mit_parse_float(_mxml_find_attr(tags[w_idx], "value")))
        if h_idx >= 0:
            film_h = Int32(_mit_parse_float(_mxml_find_attr(tags[h_idx], "value")))
        # Mitsuba's crop window is a pixel-space sub-rectangle of the full
        # film; gonzales's own crop_x0/x1/y0/y1 (pbrt "cropwindow") is the
        # same concept in NORMALIZED [0,1] fractions of the full frame --
        # convert directly.
        var cox_idx = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "crop_offset_x")
        var coy_idx = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "crop_offset_y")
        var cw_idx  = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "crop_width")
        var ch_idx  = _mit_find_child_by_attr(tags, film_idx, film_end, "integer", "name", "crop_height")
        if cox_idx >= 0 or coy_idx >= 0 or cw_idx >= 0 or ch_idx >= 0:
            var cox = _mit_parse_float(_mxml_find_attr(tags[cox_idx], "value")) if cox_idx >= 0 else Float32(0)
            var coy = _mit_parse_float(_mxml_find_attr(tags[coy_idx], "value")) if coy_idx >= 0 else Float32(0)
            var cw  = _mit_parse_float(_mxml_find_attr(tags[cw_idx], "value"))  if cw_idx  >= 0 else Float32(film_w)
            var ch  = _mit_parse_float(_mxml_find_attr(tags[ch_idx], "value"))  if ch_idx  >= 0 else Float32(film_h)
            s_ptr[0].crop_x0 = cox / Float32(film_w)
            s_ptr[0].crop_x1 = (cox + cw) / Float32(film_w)
            s_ptr[0].crop_y0 = coy / Float32(film_h)
            s_ptr[0].crop_y1 = (coy + ch) / Float32(film_h)
    s_ptr[0].film_w = film_w
    s_ptr[0].film_h = film_h
    s_ptr[0].camera_fov = _mit_shorter_axis_fov(fov_val, fov_axis, film_w, film_h)

    var sampler_idx = _mit_find_child(tags, start, end, "sampler")
    if sampler_idx >= 0:
        var sampler_end = _mit_block_end(tags, sampler_idx)
        var spp_idx = _mit_find_child_by_attr(tags, sampler_idx, sampler_end, "integer", "name", "sampleCount")
        if spp_idx < 0:
            spp_idx = _mit_find_child_by_attr(tags, sampler_idx, sampler_end, "integer", "name", "sample_count")
        if spp_idx >= 0:
            s_ptr[0].samples_per_pixel = Int32(_mit_parse_float(_mxml_find_attr(tags[spp_idx], "value")))

# ── BSDF -> NamedMaterial ────────────────────────────────────────────────────

def _mit_build_named_material(tags: List[MitsubaTag], open_idx: Int, end: Int,
                              name: String, mtype: String,
                              s_ptr: UnsafePointer[SceneParseState, MutExternalOrigin]) -> NamedMaterial:
    # Wrapper bsdfs (normalmap/bumpmap/twosided) carry the real material as
    # one nested <bsdf> child -- unwrap to it and use ITS type/params.
    # Two-sidedness itself still isn't modeled (v1 scope), same
    # "approximate, don't silently drop" philosophy as elsewhere in this
    # parser (e.g. the dielectric alpha-as-is handling below) -- but
    # "normalmap"/"bumpmap" DO now carry their perturbation through (see
    # below), unlike the original v1 unwrap-and-discard.
    var eff_start = open_idx
    var eff_end = end
    var eff_type = mtype
    var normal_tex_idx_for_mat = Int32(-1)
    if eff_type == "normalmap" or eff_type == "bumpmap":
        var nm_fname_idx = _mit_find_child_by_attr(tags, open_idx, end, "string", "name", "filename")
        if nm_fname_idx >= 0:
            var nm_file = s_ptr[0].scene_dir + _mxml_find_attr(tags[nm_fname_idx], "value")
            normal_tex_idx_for_mat = Int32(len(s_ptr[0].tex_names))
            s_ptr[0].tex_names.append(String("__normalmap"))
            s_ptr[0].tex_files.append(nm_file)
        var nested_idx = _mit_find_child(tags, open_idx + 1, end, "bsdf")
        if nested_idx >= 0:
            eff_start = nested_idx
            eff_end = _mit_block_end(tags, nested_idx)
            eff_type = _mxml_find_attr(tags[nested_idx], "type")
    elif eff_type == "twosided":
        var nested_idx = _mit_find_child(tags, open_idx + 1, end, "bsdf")
        if nested_idx >= 0:
            eff_start = nested_idx
            eff_end = _mit_block_end(tags, nested_idx)
            eff_type = _mxml_find_attr(tags[nested_idx], "type")

    var nm = NamedMaterial(name)
    nm.normal_tex_idx = normal_tex_idx_for_mat
    if eff_type == "diffuse":
        nm.kind = MatKind.diffuse
        var refl_idx = _mit_find_value_tag(tags, eff_start, eff_end, "reflectance")
        if refl_idx >= 0 and _mxml_find_attr(tags[refl_idx], "type") == "bitmap":
            var refl_end = _mit_block_end(tags, refl_idx)
            var bmp_fname_idx = _mit_find_child_by_attr(tags, refl_idx, refl_end, "string", "name", "filename")
            if bmp_fname_idx >= 0:
                var bmp_file = s_ptr[0].scene_dir + _mxml_find_attr(tags[bmp_fname_idx], "value")
                nm.tex_idx = Int32(len(s_ptr[0].tex_names))
                s_ptr[0].tex_names.append(String("__mitsuba_bitmap"))
                s_ptr[0].tex_files.append(bmp_file)
                var to_uv_idx = _mit_find_child_by_attr(tags, refl_idx, refl_end, "transform", "name", "to_uv")
                if to_uv_idx >= 0:
                    var to_uv_end = _mit_block_end(tags, to_uv_idx)
                    var scale_idx = _mit_find_child(tags, to_uv_idx, to_uv_end, "scale")
                    if scale_idx >= 0:
                        var sx = _mxml_find_attr(tags[scale_idx], "x")
                        var sy = _mxml_find_attr(tags[scale_idx], "y")
                        if sx != "":
                            nm.tex_uscale = _mit_parse_float(sx)
                        if sy != "":
                            nm.tex_vscale = _mit_parse_float(sy)
                        elif sx != "":
                            nm.tex_vscale = nm.tex_uscale
        elif refl_idx >= 0:
            var rf = _mit_parse_floats(_mxml_find_attr(tags[refl_idx], "value"))
            if len(rf) >= 3:
                nm.albedo = RGB(rf[0], rf[1], rf[2])
            elif len(rf) == 1:
                nm.albedo = RGB(rf[0])
    elif eff_type == "dielectric" or eff_type == "thindielectric" or eff_type == "roughdielectric":
        nm.kind = MatKind.thin_dielectric if eff_type == "thindielectric" else MatKind.dielectric
        var int_idx = _mit_find_child_by_attr(tags, eff_start, eff_end, "float", "name", "intIOR")
        var ext_idx = _mit_find_child_by_attr(tags, eff_start, eff_end, "float", "name", "extIOR")
        var int_ior = _mit_parse_float(_mxml_find_attr(tags[int_idx], "value")) if int_idx >= 0 else Float32(1.5)
        var ext_ior = _mit_parse_float(_mxml_find_attr(tags[ext_idx], "value")) if ext_idx >= 0 else Float32(1.0)
        nm.ior = int_ior / ext_ior
        # Mitsuba's "alpha" IS the microfacet alpha already (no perceptual-
        # roughness remap, unlike pbrt's remaproughness=true default) -- use
        # it as-is, matching pbrt's own remaproughness=false code path.
        var alpha_idx = _mit_find_child_by_attr(tags, eff_start, eff_end, "float", "name", "alpha")
        if alpha_idx >= 0:
            var a = _mit_parse_float(_mxml_find_attr(tags[alpha_idx], "value"))
            nm.roughness_u = a
            nm.roughness_v = a
    else:
        print("Warning: unsupported Mitsuba bsdf type '" + eff_type + "' -- rendering as flat 50%-grey diffuse. Supported in this parser: diffuse, dielectric.")
        nm.kind = MatKind.diffuse
    return nm^

# ── Shape (serialized / rectangle) ──────────────────────────────────────────

def _mit_resolve_material_idx(tags: List[MitsubaTag], shape_idx: Int, end: Int,
                              s_ptr: UnsafePointer[SceneParseState, MutExternalOrigin]) -> Int32:
    var mat_idx_result = Int32(-1)
    var inline_bsdf_idx = _mit_find_child(tags, shape_idx, end, "bsdf")
    if inline_bsdf_idx >= 0:
        var ib_end = _mit_block_end(tags, inline_bsdf_idx)
        var mtype = _mxml_find_attr(tags[inline_bsdf_idx], "type")
        var nm = _mit_build_named_material(tags, inline_bsdf_idx, ib_end, String(""), mtype, s_ptr)
        s_ptr[0].named_materials.append(nm^)
        mat_idx_result = Int32(len(s_ptr[0].named_materials) - 1)
    else:
        var ref_idx = _mit_find_child_by_attr(tags, shape_idx, end, "ref", "name", "bsdf")
        if ref_idx >= 0:
            var ref_id = _mxml_find_attr(tags[ref_idx], "id")
            for k in range(len(s_ptr[0].named_materials)):
                if s_ptr[0].named_materials[k].name == ref_id:
                    mat_idx_result = Int32(k)
                    break
    return mat_idx_result

def _mit_resolve_emitter(tags: List[MitsubaTag], shape_idx: Int, end: Int) -> Tuple[Bool, RGB]:
    var em_idx = _mit_find_child(tags, shape_idx, end, "emitter")
    if em_idx < 0:
        return (False, RGB(Float32(0)))
    var em_end = _mit_block_end(tags, em_idx)
    var rad_idx = _mit_find_value_tag(tags, em_idx, em_end, "radiance")
    var rad = RGB(Float32(1))
    if rad_idx >= 0:
        var rf = _mit_parse_floats(_mxml_find_attr(tags[rad_idx], "value"))
        if len(rf) >= 3:
            rad = RGB(rf[0], rf[1], rf[2])
        elif len(rf) == 1:
            rad = RGB(rf[0])
    return (True, rad)

def _mit_process_sphere(tags: List[MitsubaTag], shape_idx: Int, end: Int,
                        s_ptr: UnsafePointer[SceneParseState, MutExternalOrigin]):
    """Native analytic Sphere_C for Mitsuba's built-in `sphere` shape --
    exact intersection and exact shading (see shading.mojo's
    primId.type==4 branches in shade_dielectric/shade_thin_dielectric/
    shade_conductor/_build_geom_context_full), no tessellation. Mirrors
    pbrt_parser.mojo's handle_sphere_shape: appends to the SceneParseState's
    flat spheres_* arrays instead of calling store_mesh.

    `<point name="center"/>`/`<float name="radius"/>` are OBJECT-space
    values (matching pbrt's own sphere radius convention); `s_ptr[0].ctm`
    (already set by the caller from the shape's own <transform>, if any)
    is then applied in full to the center (translation+rotation+scale,
    more general than pbrt's translation-only sphere since Mitsuba allows
    a non-origin object-space center) and as a uniform-scale factor to the
    radius. Non-uniform scale would turn a sphere into an ellipsoid, which
    Sphere_C can't represent -- out of scope, not used by any scene seen
    so far."""
    var s_center_obj = Vec3f(Float32(0), Float32(0), Float32(0))
    var center_idx = _mit_find_child_by_attr(tags, shape_idx, end, "point", "name", "center")
    if center_idx >= 0:
        s_center_obj = _mit_parse_vec3_attr(tags[center_idx], "value")
    var s_radius = Float32(1)
    var radius_idx = _mit_find_child_by_attr(tags, shape_idx, end, "float", "name", "radius")
    if radius_idx >= 0:
        s_radius = _mit_parse_float(_mxml_find_attr(tags[radius_idx], "value"))

    var ctm = s_ptr[0].ctm.copy()
    var cx = ctm[0]*s_center_obj[0] + ctm[4]*s_center_obj[1] + ctm[8]*s_center_obj[2]  + ctm[12]
    var cy = ctm[1]*s_center_obj[0] + ctm[5]*s_center_obj[1] + ctm[9]*s_center_obj[2]  + ctm[13]
    var cz = ctm[2]*s_center_obj[0] + ctm[6]*s_center_obj[1] + ctm[10]*s_center_obj[2] + ctm[14]
    var sx = sqrt(ctm[0]*ctm[0] + ctm[1]*ctm[1] + ctm[2]*ctm[2])
    if sx < Float32(1e-6):
        sx = Float32(1.0)
    var radius = s_radius * sx

    var mat_idx_result = _mit_resolve_material_idx(tags, shape_idx, end, s_ptr)
    var em = _mit_resolve_emitter(tags, shape_idx, end)

    s_ptr[0].spheres_cx.append(cx)
    s_ptr[0].spheres_cy.append(cy)
    s_ptr[0].spheres_cz.append(cz)
    s_ptr[0].spheres_r.append(radius)
    s_ptr[0].spheres_mat.append(mat_idx_result)
    s_ptr[0].spheres_inside_med.append(Int32(-1))
    s_ptr[0].spheres_outside_med.append(Int32(-1))
    s_ptr[0].spheres_al.append(em[0])
    s_ptr[0].spheres_rgb.append(em[1])

def _mit_process_shape(tags: List[MitsubaTag], shape_idx: Int, end: Int,
                       s_ptr: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var shape_type = _mxml_find_attr(tags[shape_idx], "type")

    var tf_idx = _mit_find_transform(tags, shape_idx, end)
    if tf_idx >= 0:
        var tf_end = _mit_block_end(tags, tf_idx)
        s_ptr[0].ctm = _mit_matrix_rowmajor_to_ctm(_mit_parse_transform_block(tags, tf_idx, tf_end))
    else:
        s_ptr[0].ctm = _mit_identity_ctm()

    if shape_type == "sphere":
        # Native analytic Sphere_C (see _mit_process_sphere) -- appends to
        # SceneParseState's spheres_* arrays directly, not store_mesh, so
        # this returns before the tmp_f/tmp_i/nv/nt declarations shared by
        # the serialized/rectangle branches below even come into scope,
        # matching the structure of every other early-return case here.
        _mit_process_sphere(tags, shape_idx, end, s_ptr)
        return

    var tmp_f: UnsafePointer[Float32, MutExternalOrigin]
    var tmp_i: UnsafePointer[Int32, MutExternalOrigin]
    var nv: Int32
    var nt: Int32
    var uvs = List[Float32]()
    var normals = List[Float32]()

    if shape_type == "serialized":
        var fname_idx = _mit_find_child_by_attr(tags, shape_idx, end, "string", "name", "filename")
        if fname_idx < 0:
            print("Warning: Mitsuba 'serialized' shape with no filename -- skipped.")
            return
        var full_path = s_ptr[0].scene_dir + _mxml_find_attr(tags[fname_idx], "value")
        var mesh = load_mitsuba_serialized(full_path)
        if mesh.n_verts == Int32(0):
            print("Warning: failed to load Mitsuba mesh:", full_path)
            return
        nv = mesh.n_verts
        nt = mesh.n_tris
        tmp_f = alloc[Float32](Int(nv) * 3)
        for k in range(Int(nv) * 3):
            tmp_f[k] = mesh.positions[k]
        tmp_i = alloc[Int32](Int(nt) * 3)
        for k in range(Int(nt) * 3):
            tmp_i[k] = mesh.indices[k]
        uvs = mesh.uvs.copy()
        normals = mesh.normals.copy()
    elif shape_type == "rectangle":
        # Mitsuba's built-in rectangle: unit square in the object-space XY
        # plane (z=0), spanning [-1,1] x [-1,1], normal +Z -- no external
        # mesh file, just a hardcoded 4-vertex/2-triangle quad.
        nv = Int32(4)
        nt = Int32(2)
        tmp_f = alloc[Float32](12)
        tmp_f[0]  = Float32(-1); tmp_f[1]  = Float32(-1); tmp_f[2]  = Float32(0)
        tmp_f[3]  = Float32(1);  tmp_f[4]  = Float32(-1); tmp_f[5]  = Float32(0)
        tmp_f[6]  = Float32(1);  tmp_f[7]  = Float32(1);  tmp_f[8]  = Float32(0)
        tmp_f[9]  = Float32(-1); tmp_f[10] = Float32(1);  tmp_f[11] = Float32(0)
        tmp_i = alloc[Int32](6)
        tmp_i[0] = Int32(0); tmp_i[1] = Int32(1); tmp_i[2] = Int32(2)
        tmp_i[3] = Int32(0); tmp_i[4] = Int32(2); tmp_i[5] = Int32(3)
        # Natural unit-square UVs matching the vertex order above -- scaled
        # below (once the material's bitmap `to_uv` scale, if any, is known)
        # for a tiled floor texture like sphere_sms.xml's own.
        uvs.append(Float32(0)); uvs.append(Float32(0))
        uvs.append(Float32(1)); uvs.append(Float32(0))
        uvs.append(Float32(1)); uvs.append(Float32(1))
        uvs.append(Float32(0)); uvs.append(Float32(1))
    else:
        print("Warning: unsupported Mitsuba shape type '" + shape_type + "' -- skipped. Supported in this parser: serialized, rectangle.")
        return

    var mat_idx_result = _mit_resolve_material_idx(tags, shape_idx, end, s_ptr)
    s_ptr[0].cur_attr.mat_idx = mat_idx_result
    var em = _mit_resolve_emitter(tags, shape_idx, end)
    s_ptr[0].cur_attr.is_alight = em[0]
    s_ptr[0].cur_attr.al_rgb = em[1]

    if mat_idx_result >= Int32(0) and len(uvs) > 0:
        var mnm = s_ptr[0].named_materials[Int(mat_idx_result)]
        if mnm.tex_uscale != Float32(1) or mnm.tex_vscale != Float32(1):
            for k in range(len(uvs) // 2):
                uvs[k*2]   *= mnm.tex_uscale
                uvs[k*2+1] *= mnm.tex_vscale

    store_mesh(s_ptr, tmp_f, tmp_i, nv, nt)
    tmp_f.free()
    tmp_i.free()

    var last = len(s_ptr[0].meshes) - 1
    if len(uvs) > 0:
        s_ptr[0].meshes[last].uvs.reserve(len(uvs))
        for k in range(len(uvs)):
            s_ptr[0].meshes[last].uvs.append(uvs[k])

    if len(normals) > 0:
        var nrm_obj = alloc[Float32](Int(nv) * 3)
        for k in range(Int(nv) * 3):
            nrm_obj[k] = normals[k]
        var ctm_inv = alloc[Float32](16)
        _ = matrix_invert(s_ptr[0].ctm.unsafe_ptr(), ctm_inv)
        var nrm_world = alloc[Float32](Int(nv) * 3)
        transform_normals(ctm_inv, nrm_obj, nv, nrm_world)
        ref last_mesh = s_ptr[0].meshes[last]
        last_mesh.normals.reserve(Int(nv) * 3)
        for ni in range(Int(nv)):
            var nx = nrm_world[ni * 3 + 0]; var ny = nrm_world[ni * 3 + 1]; var nz = nrm_world[ni * 3 + 2]
            var nlen = sqrt(nx * nx + ny * ny + nz * nz)
            if nlen > Float32(1e-12):
                var invn = Float32(1) / nlen
                nx *= invn; ny *= invn; nz *= invn
            last_mesh.normals.append(nx)
            last_mesh.normals.append(ny)
            last_mesh.normals.append(nz)
        nrm_world.free(); ctm_inv.free(); nrm_obj.free()

# ── Top-level entry point ────────────────────────────────────────────────────

def mojo_parse_mitsuba_scene(path: UnsafePointer[UInt8, MutExternalOrigin],
                             verbose: Bool = False,
                            ) -> UnsafePointer[ParsedScene_Mojo, MutExternalOrigin]:
    # mojo_parse_scene (pbrt_parser.mojo) does this at its own entry --
    # needed once, globally, before any OIIO `texture()` bridge call
    # (shading.mojo's sample_texture CPU branch) or every lookup
    # dereferences a null TextureSystem singleton and segfaults. Missing
    # here meant any Mitsuba scene with a real bitmap texture (e.g. a
    # tiled floor) crashed the moment it was first sampled.
    external_call["createTextureSystem", NoneType]()
    var path_str = String(unsafe_from_utf8_ptr=path.as_immutable())

    var pi = 0
    while path[pi] != UInt8(0):
        pi += 1
    var last_slash = -1
    for ki in range(pi):
        if path[ki] == UInt8(47):
            last_slash = ki
    var scene_dir = String("")
    if last_slash >= 0:
        var dir_tmp = alloc[UInt8](last_slash + 2)
        for ki in range(last_slash + 1):
            dir_tmp[ki] = path[ki]
        dir_tmp[last_slash + 1] = UInt8(0)
        scene_dir = String(unsafe_from_utf8_ptr=dir_tmp.as_immutable())
        dir_tmp.free()

    var byte_list: List[UInt8]
    try:
        var fh = open(path_str, "r")
        byte_list = fh.read_bytes()
        fh.close()
    except:
        print("Error: cannot open scene file:", path_str)
        return UnsafePointer[ParsedScene_Mojo, MutExternalOrigin].unsafe_dangling()

    var n = len(byte_list)
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = byte_list[i]
    var tags = tokenize_mitsuba_xml(buf, n)
    buf.free()
    _mit_apply_defaults(tags)

    var scene_idx = -1
    for i in range(len(tags)):
        if not tags[i].is_close and tags[i].name == "scene":
            scene_idx = i
            break
    if scene_idx < 0:
        print("Error: no <scene> element found in", path_str)
        return UnsafePointer[ParsedScene_Mojo, MutExternalOrigin].unsafe_dangling()

    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    s_ptr[0].scene_dir = scene_dir

    var scene_end = _mit_block_end(tags, scene_idx)
    var i = scene_idx + 1
    while i < scene_end - 1:
        ref t = tags[i]
        if t.is_close:
            i += 1
            continue
        var blk_end = _mit_block_end(tags, i)
        if t.name == "sensor":
            _mit_process_sensor(tags, i, blk_end, s_ptr)
        elif t.name == "bsdf":
            var nm = _mit_build_named_material(tags, i, blk_end, _mxml_find_attr(t, "id"), _mxml_find_attr(t, "type"), s_ptr)
            s_ptr[0].named_materials.append(nm^)
        elif t.name == "shape":
            _mit_process_shape(tags, i, blk_end, s_ptr)
        i = blk_end

    var psc = alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, verbose)
    _ = s_ptr.take_pointee()
    s_ptr.free()
    return psc
