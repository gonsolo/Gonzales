from std.ffi import external_call
from std.time import perf_counter_ns
from std.memory import alloc
from std.math import tan, sqrt, abs
from .lexer import (PbrtScanner, scanner_open, scanner_free, scanner_is_at_end,
                    scanner_scan_token, scanner_parse_quoted_string, scanner_parse_param_header,
                    scanner_scan_char, scanner_scan_float, scanner_scan_floats,
                    scanner_scan_int, scanner_scan_ints,
                    _psc_streq, _psc_type_is_float, _psc_type_is_int, _psc_type_is_str,
                    _psc_scan_rgb, _psc_scan_one_float, _psc_scan_one_int, _psc_scan_one_str,
                    _psc_skip_value, _psc_skip_params, _psc_skip_line)
from .parse_types import (SceneParseState, MeshAccum, NamedMaterial,
                           ctm_push, ctm_pop, PSC_NAME_MAX, PSC_FILE_MAX,
                           HAIR_EVAL_N)
from .geometry import (RGB, SampledSpectrum, Point3f, Vec3f, Material_C, AreaLight_C,
                        Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C,
                        TriangleMesh_C, PrimId_C, Medium_C, MediumInterface_C, PI,
                        LightSampler_C)
from .transform import matrix_multiply, matrix_invert, transform_points, transform_normals
from .bvh import BVH2Node, SceneDescriptor2_C, build_bvh2
from .sampling import gaussian_norm
from .ply import load_ply
from .material_builder import _psc_handle_make_named_material, _psc_handle_named_material
from .light_builder import _psc_handle_area_light_source, handle_light_source

# ── Output struct ─────────────────────────────────────────────────────────────

struct ParsedScene_Mojo:
    var raster_to_camera: UnsafePointer[Float32, MutAnyOrigin]   # 16 floats, column-major
    var camera_to_world:  UnsafePointer[Float32, MutAnyOrigin]   # 16 floats, column-major
    var materials:        UnsafePointer[Material_C, MutAnyOrigin]
    var material_count:   Int32
    var area_lights:      UnsafePointer[AreaLight_C, MutAnyOrigin]
    var area_light_count: Int32
    var meshes:           UnsafePointer[TriangleMesh_C, MutAnyOrigin]
    var mesh_pts:         UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var mesh_vis:         UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_fis:         UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_n_verts:     UnsafePointer[Int32, MutAnyOrigin]
    var mesh_n_tris:      UnsafePointer[Int32, MutAnyOrigin]
    var mesh_uv_n_verts:  UnsafePointer[Int32, MutAnyOrigin]  # per-mesh UV vertex count; 0 = no UVs
    var mesh_nrm_n_verts: UnsafePointer[Int32, MutAnyOrigin]  # per-mesh normal vertex count; 0 = no shading normals
    var mesh_count:       Int32
    var bvh_nodes:        UnsafePointer[BVH2Node, MutAnyOrigin]
    var prim_ids:         UnsafePointer[PrimId_C, MutAnyOrigin]
    var bvh_node_count:   Int32
    var prim_count:       Int32
    var film_w:           Int32
    var film_h:           Int32
    var camera_fov:       Float32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    UnsafePointer[UInt8, MutAnyOrigin]      # null-terminated
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_norm_x:    Float32
    var filter_norm_y:    Float32
    var filter_weight:    Float32
    var filter_type:      Int32
    var samples_per_pixel: Int32
    var log2_spp:         Int32
    var n_base4_digits:   Int32
    var max_depth:        Int32
    var rng_seed:         UInt64
    var tex_filenames:    UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    var tex_count:        Int32
    var distant_lights:   UnsafePointer[DistantLight_C, MutAnyOrigin]
    var distant_count:    Int32
    var point_lights:     UnsafePointer[PointLight_C, MutAnyOrigin]
    var point_count:      Int32
    var infinite_lights:  UnsafePointer[InfiniteLight_C, MutAnyOrigin]
    var infinite_count:   Int32
    var spheres:          UnsafePointer[Sphere_C, MutAnyOrigin]
    var sphere_count:     Int32
    var mediums:          UnsafePointer[Medium_C, MutAnyOrigin]
    var medium_count:     Int32
    var medium_ifaces:    UnsafePointer[MediumInterface_C, MutAnyOrigin]
    var medium_iface_count: Int32
    var light_sampler:    LightSampler_C

# ── Matrix utilities ──────────────────────────────────────────────────────────

def _psc_identity(m: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        m[i] = Float32(0)
    m[0] = Float32(1)
    m[5] = Float32(1)
    m[10] = Float32(1)
    m[15] = Float32(1)

def _psc_matcopy(dst: UnsafePointer[Float32, MutAnyOrigin],
                src: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        dst[i] = src[i]

def _psc_ctm_concat(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                   t: UnsafePointer[Float32, MutAnyOrigin]):
    """Compute s.ctm = s.ctm × t and store back."""
    var result = alloc[Float32](16)
    matrix_multiply(s[0].ctm.unsafe_ptr(), t, result)
    for i in range(16):
        s[0].ctm[i] = result[i]
    result.free()

def _psc_row_to_col(col_out: UnsafePointer[Float32, MutAnyOrigin],
                   row_in:  UnsafePointer[Float32, MutAnyOrigin]):
    for row in range(4):
        for col in range(4):
            col_out[col * 4 + row] = row_in[row * 4 + col]

# ── Transform keyword handlers ────────────────────────────────────────────────

def _psc_handle_translate(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                         s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Translate tx ty tz  →  CTM = CTM × T(tx,ty,tz)"""
    var v = alloc[Float32](3)
    v[0] = Float32(0); v[1] = Float32(0); v[2] = Float32(0)
    _ = scanner_scan_float(handle, v + 0)
    _ = scanner_scan_float(handle, v + 1)
    _ = scanner_scan_float(handle, v + 2)
    var t = alloc[Float32](16)
    _psc_identity(t)
    t[12] = v[0]; t[13] = v[1]; t[14] = v[2]   # col-major: col3 = (tx,ty,tz,1)
    _psc_ctm_concat(s, t)
    v.free(); t.free()

def _psc_handle_scale_kw(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                        s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Scale sx sy sz  →  CTM = CTM × S(sx,sy,sz)"""
    var v = alloc[Float32](3)
    v[0] = Float32(1); v[1] = Float32(1); v[2] = Float32(1)
    _ = scanner_scan_float(handle, v + 0)
    _ = scanner_scan_float(handle, v + 1)
    _ = scanner_scan_float(handle, v + 2)
    var t = alloc[Float32](16)
    _psc_identity(t)
    t[0] = v[0]; t[5] = v[1]; t[10] = v[2]     # col-major: diagonal
    _psc_ctm_concat(s, t)
    v.free(); t.free()

def _psc_handle_rotate(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Rotate angle ax ay az  →  CTM = CTM × R(angle, axis)"""
    from std.math import sin as _sin, cos as _cos, sqrt as _sqrt
    var rv = alloc[Float32](4)  # angle, ax, ay, az
    rv[0] = Float32(0); rv[1] = Float32(0); rv[2] = Float32(0); rv[3] = Float32(1)
    _ = scanner_scan_float(handle, rv + 0)
    _ = scanner_scan_float(handle, rv + 1)
    _ = scanner_scan_float(handle, rv + 2)
    _ = scanner_scan_float(handle, rv + 3)
    var angle = rv[0] * PI / Float32(180)
    var ax = rv[1]; var ay = rv[2]; var az = rv[3]
    var ln = _sqrt(ax*ax + ay*ay + az*az)
    if ln > Float32(1e-12): ax /= ln; ay /= ln; az /= ln
    var c = _cos(angle); var sv = _sin(angle); var mc = Float32(1) - c
    var t = alloc[Float32](16)
    # Column-major rotation matrix (standard Rodrigues)
    t[0]  = c + ax*ax*mc;       t[1]  = ay*ax*mc + az*sv;   t[2]  = az*ax*mc - ay*sv;   t[3]  = Float32(0)
    t[4]  = ax*ay*mc - az*sv;   t[5]  = c + ay*ay*mc;       t[6]  = az*ay*mc + ax*sv;   t[7]  = Float32(0)
    t[8]  = ax*az*mc + ay*sv;   t[9]  = ay*az*mc - ax*sv;   t[10] = c + az*az*mc;        t[11] = Float32(0)
    t[12] = Float32(0);         t[13] = Float32(0);          t[14] = Float32(0);          t[15] = Float32(1)
    _psc_ctm_concat(s, t)
    rv.free(); t.free()

def _psc_handle_lookat(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """LookAt ex ey ez  lx ly lz  ux uy uz"""
    from std.math import sqrt as _sqrt
    var v = alloc[Float32](9)
    for i in range(9): _ = scanner_scan_float(handle, v + i)
    var ex = v[0]; var ey = v[1]; var ez = v[2]
    var lx = v[3]; var ly = v[4]; var lz = v[5]
    var ux = v[6]; var uy = v[7]; var uz = v[8]
    v.free()

    var dx = lx - ex; var dy = ly - ey; var dz = lz - ez
    var dl = _sqrt(dx*dx + dy*dy + dz*dz)
    if dl > Float32(1e-12): dx /= dl; dy /= dl; dz /= dl

    var ul = _sqrt(ux*ux + uy*uy + uz*uz)
    if ul > Float32(1e-12): ux /= ul; uy /= ul; uz /= ul
    var rx = uy*dz - uz*dy
    var ry = uz*dx - ux*dz
    var rz = ux*dy - uy*dx
    var rl = _sqrt(rx*rx + ry*ry + rz*rz)
    if rl > Float32(1e-12): rx /= rl; ry /= rl; rz /= rl

    var nx = dy*rz - dz*ry
    var ny = dz*rx - dx*rz
    var nz = dx*ry - dy*rx

    var t = alloc[Float32](16)
    t[0]  = rx;  t[1]  = nx;  t[2]  = dx;  t[3]  = Float32(0)
    t[4]  = ry;  t[5]  = ny;  t[6]  = dy;  t[7]  = Float32(0)
    t[8]  = rz;  t[9]  = nz;  t[10] = dz;  t[11] = Float32(0)
    t[12] = -(rx*ex + ry*ey + rz*ez)
    t[13] = -(nx*ex + ny*ey + nz*ez)
    t[14] = -(dx*ex + dy*ey + dz*ez)
    t[15] = Float32(1)
    _psc_ctm_concat(s, t)
    t.free()

# ── Directive handlers ────────────────────────────────────────────────────────

def _psc_handle_integrator(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                          s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "maxdepth") and _psc_type_is_int(type_buf):
            s[0].max_depth = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_sampler(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                       s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if (_psc_streq(name_buf, "pixelsamples") or _psc_streq(name_buf, "samples")) and _psc_type_is_int(type_buf):
            s[0].samples_per_pixel = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_filter(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    if _psc_streq(sbuf, "triangle") or _psc_streq(sbuf, "tent"):
        s[0].filter_type = Int32(1)
    elif _psc_streq(sbuf, "box"):
        s[0].filter_type = Int32(2)
    else:
        s[0].filter_type = Int32(0)  # gaussian (default)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "xradius") and _psc_type_is_float(type_buf):
            s[0].filter_support_x = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "yradius") and _psc_type_is_float(type_buf):
            s[0].filter_support_y = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "sigma") and _psc_type_is_float(type_buf):
            s[0].filter_sigma = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_film(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                    s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "xresolution") and _psc_type_is_int(type_buf):
            s[0].film_w = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "yresolution") and _psc_type_is_int(type_buf):
            s[0].film_h = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            var fn_tmp = alloc[UInt8](PSC_FILE_MAX)
            _psc_scan_one_str(handle, fn_tmp, Int32(PSC_FILE_MAX), is_array)
            s[0].film_filename = String(unsafe_from_utf8_ptr=fn_tmp.as_immutable())
            fn_tmp.free()
        elif _psc_streq(name_buf, "iso") and _psc_type_is_float(type_buf):
            s[0].film_iso = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "maxcomponentvalue") and _psc_type_is_float(type_buf):
            s[0].film_max_comp = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_camera(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    # Copy current CTM into cam2w_raw
    for i in range(16): s[0].cam2w_raw[i] = s[0].ctm[i]
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "fov") and _psc_type_is_float(type_buf):
            s[0].camera_fov = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_transform(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                         s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    _ = scanner_scan_char(handle, UInt8(91))  # '['
    var tmp = alloc[Float32](1)
    for i in range(16):
        _ = scanner_scan_float(handle, tmp)
        s[0].ctm[i] = tmp[0]
    tmp.free()
    _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_handle_world_begin(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    for i in range(16): s[0].ctm[i] = Float32(0)
    s[0].ctm[0] = Float32(1); s[0].ctm[5] = Float32(1)
    s[0].ctm[10] = Float32(1); s[0].ctm[15] = Float32(1)
    s[0].ctm_stack.clear()

def _psc_handle_attribute_begin(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    ctm_push(s[0])
    s[0].attr_stack.append(s[0].cur_attr)

def _psc_handle_attribute_end(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    ctm_pop(s[0])
    if len(s[0].attr_stack) > 0:
        s[0].cur_attr = s[0].attr_stack[len(s[0].attr_stack) - 1]
        _ = s[0].attr_stack.pop()

# ── Mesh accumulation ─────────────────────────────────────────────────────────

def store_mesh(
    s:       UnsafePointer[SceneParseState, MutAnyOrigin],
    tmp_f:   UnsafePointer[Float32, MutAnyOrigin],
    tmp_i:   UnsafePointer[Int32, MutAnyOrigin],
    n_verts: Int32,
    n_tris:  Int32,
):
    var raw_pts = alloc[Float32](Int(n_verts) * 4)
    for v in range(Int(n_verts)):
        raw_pts[v*4+0] = tmp_f[v*3+0]
        raw_pts[v*4+1] = tmp_f[v*3+1]
        raw_pts[v*4+2] = tmp_f[v*3+2]
        raw_pts[v*4+3] = Float32(1)
    var fin_pts = alloc[Float32](Int(n_verts) * 4)
    transform_points(s[0].ctm.unsafe_ptr(), raw_pts, n_verts, fin_pts)
    raw_pts.free()
    var ma = MeshAccum(
        s[0].cur_attr.mat_idx,
        s[0].cur_attr.inside_medium,
        s[0].cur_attr.outside_medium,
    )
    ma.is_area_light = s[0].cur_attr.is_alight
    ma.al_rgb = s[0].cur_attr.al_rgb
    ma.points.reserve(Int(n_verts) * 4)
    for v in range(Int(n_verts) * 4):
        ma.points.append(fin_pts[v])
    fin_pts.free()
    ma.vert_idxs.reserve(Int(n_tris) * 3)
    ma.face_idxs.reserve(Int(n_tris))
    var rev = s[0].cur_attr.reverse_orient
    for t in range(Int(n_tris)):
        ma.vert_idxs.append(Int64(tmp_i[t*3+0]))
        if rev:
            ma.vert_idxs.append(Int64(tmp_i[t*3+2]))
            ma.vert_idxs.append(Int64(tmp_i[t*3+1]))
        else:
            ma.vert_idxs.append(Int64(tmp_i[t*3+1]))
            ma.vert_idxs.append(Int64(tmp_i[t*3+2]))
        ma.face_idxs.append(Int64(3))
    s[0].meshes.append(ma^)

# ── Hair curve helpers ────────────────────────────────────────────────────────

def ensure_hair_buffer(s: UnsafePointer[SceneParseState, MutAnyOrigin]) -> Bool:
    if s[0].hair:
        return True
    s[0].hair = MeshAccum(
        s[0].cur_attr.mat_idx,
        s[0].cur_attr.inside_medium,
        s[0].cur_attr.outside_medium,
    )
    return True

def bspline3_eval(cp: UnsafePointer[Float32, MutAnyOrigin], n_cp: Int,
                  u: Float32, pt_out: UnsafePointer[Float32, MutAnyOrigin]):
    var n_seg = n_cp - 3
    if n_seg <= 0:
        pt_out[0] = cp[0]; pt_out[1] = cp[1]; pt_out[2] = cp[2]; return
    var s_f = u * Float32(n_seg)
    var seg = Int(s_f)
    if seg >= n_seg: seg = n_seg - 1
    var t = s_f - Float32(seg)
    var t2 = t * t
    var t3 = t2 * t
    var b0 = (Float32(1) - Float32(3)*t + Float32(3)*t2 - t3) / Float32(6)
    var b1 = (Float32(4) - Float32(6)*t2 + Float32(3)*t3) / Float32(6)
    var b2 = (Float32(1) + Float32(3)*t + Float32(3)*t2 - Float32(3)*t3) / Float32(6)
    var b3 = t3 / Float32(6)
    for k in range(3):
        pt_out[k] = b0*cp[seg*3+k] + b1*cp[(seg+1)*3+k] + b2*cp[(seg+2)*3+k] + b3*cp[(seg+3)*3+k]

def hair_add_quad(s: UnsafePointer[SceneParseState, MutAnyOrigin],
        p0x: Float32, p0y: Float32, p0z: Float32,
        p1x: Float32, p1y: Float32, p1z: Float32,
        ux: Float32, uy: Float32, uz: Float32, hw: Float32):
    ref h = s[0].hair.value()
    var bv = len(h.points) // 4
    h.points.append(p0x+ux*hw); h.points.append(p0y+uy*hw)
    h.points.append(p0z+uz*hw); h.points.append(Float32(1))
    h.points.append(p0x-ux*hw); h.points.append(p0y-uy*hw)
    h.points.append(p0z-uz*hw); h.points.append(Float32(1))
    h.points.append(p1x+ux*hw); h.points.append(p1y+uy*hw)
    h.points.append(p1z+uz*hw); h.points.append(Float32(1))
    h.points.append(p1x-ux*hw); h.points.append(p1y-uy*hw)
    h.points.append(p1z-uz*hw); h.points.append(Float32(1))
    h.vert_idxs.append(Int64(bv));   h.vert_idxs.append(Int64(bv+1)); h.vert_idxs.append(Int64(bv+2))
    h.face_idxs.append(Int64(3))
    h.vert_idxs.append(Int64(bv+1)); h.vert_idxs.append(Int64(bv+3)); h.vert_idxs.append(Int64(bv+2))
    h.face_idxs.append(Int64(3))

def handle_curve_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                            s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var MAX_CP = 512
    var cp_buf = alloc[Float32](MAX_CP * 3)
    var n_cp = Int32(0)
    var width = Float32(0.002)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "P"):
            var n_f = scanner_scan_floats(handle, cp_buf, Int32(MAX_CP * 3))
            n_cp = n_f / Int32(3)
            if is_array: _ = scanner_scan_char(handle, UInt8(93))
        elif _psc_type_is_float(type_buf) and (_psc_streq(name_buf, "width") or _psc_streq(name_buf, "width0")):
            width = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array: _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    type_buf.free(); name_buf.free(); ia.free()
    if n_cp < Int32(4):
        cp_buf.free(); return
    if not ensure_hair_buffer(s):
        cp_buf.free(); return
    var eval_pts = alloc[Float32](HAIR_EVAL_N * 3)
    for step in range(HAIR_EVAL_N):
        var u = Float32(step) / Float32(HAIR_EVAL_N - 1)
        bspline3_eval(cp_buf, Int(n_cp), u, eval_pts + step * 3)
    cp_buf.free()
    var raw4 = alloc[Float32](HAIR_EVAL_N * 4)
    var xfm4 = alloc[Float32](HAIR_EVAL_N * 4)
    for step in range(HAIR_EVAL_N):
        raw4[step*4+0] = eval_pts[step*3+0]
        raw4[step*4+1] = eval_pts[step*3+1]
        raw4[step*4+2] = eval_pts[step*3+2]
        raw4[step*4+3] = Float32(1)
    transform_points(s[0].ctm.unsafe_ptr(), raw4, Int32(HAIR_EVAL_N), xfm4)
    raw4.free()
    var hw = width / Float32(2)
    for seg in range(HAIR_EVAL_N - 1):
        var p0x = xfm4[seg*4+0]; var p0y = xfm4[seg*4+1]; var p0z = xfm4[seg*4+2]
        var p1x = xfm4[(seg+1)*4+0]; var p1y = xfm4[(seg+1)*4+1]; var p1z = xfm4[(seg+1)*4+2]
        var tx = p1x-p0x; var ty = p1y-p0y; var tz = p1z-p0z
        var tlen = sqrt(tx*tx + ty*ty + tz*tz)
        if tlen < Float32(1e-10): continue
        tx /= tlen; ty /= tlen; tz /= tlen
        var ux: Float32; var uy: Float32; var uz: Float32
        if abs(ty) < Float32(0.9):
            ux = -tz; uy = Float32(0); uz = tx
        else:
            ux = Float32(0); uy = tz; uz = -ty
        var ulen = sqrt(ux*ux + uy*uy + uz*uz)
        ux /= ulen; uy /= ulen; uz /= ulen
        var vx = ty*uz - tz*uy; var vy = tz*ux - tx*uz; var vz = tx*uy - ty*ux
        hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, ux,uy,uz, hw)
        hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, vx,vy,vz, hw)
    xfm4.free(); eval_pts.free()

def flush_hair(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    if not s[0].hair:
        return
    var h = s[0].hair.take()
    if len(h.face_idxs) == 0:
        return
    s[0].meshes.append(h^)

# ── Medium handlers ───────────────────────────────────────────────────────────

def handle_named_medium(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                  s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var name_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, name_buf, 64)
    var type_buf  = alloc[UInt8](64)
    var val_buf   = alloc[UInt8](64)
    var sa = alloc[Float32](3); sa[0] = Float32(0); sa[1] = Float32(0); sa[2] = Float32(0)
    var ss = alloc[Float32](3); ss[0] = Float32(0); ss[1] = Float32(0); ss[2] = Float32(0)
    var g_val = Float32(0)
    var scale = Float32(1)
    var is_hom = False
    var ia = alloc[Int32](1)
    while True:
        ia[0] = Int32(0)
        var ok = scanner_parse_param_header(handle, type_buf, Int32(64), val_buf, Int32(64), ia)
        if ok == Int32(0):
            break
        var is_arr = ia[0]
        if _psc_streq(val_buf, "type"):
            var tmp = alloc[UInt8](64)
            _ = scanner_parse_quoted_string(handle, tmp, 64)
            if _psc_streq(tmp, "homogeneous"):
                is_hom = True
            tmp.free()
        elif _psc_streq(val_buf, "sigma_a") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, sa, is_arr)
        elif _psc_streq(val_buf, "sigma_s") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, ss, is_arr)
        elif _psc_streq(val_buf, "g") and _psc_type_is_float(type_buf):
            g_val = _psc_scan_one_float(handle, is_arr)
        elif _psc_streq(val_buf, "scale") and _psc_type_is_float(type_buf):
            scale = _psc_scan_one_float(handle, is_arr)
        else:
            _psc_skip_value(handle, type_buf, is_arr)
            if is_arr:
                _ = scanner_scan_char(handle, UInt8(93))
    if is_hom:
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_immutable())
        s[0].med_names.append(name_str)
        s[0].med_sa.append(sa[0] * scale)
        s[0].med_sa.append(sa[1] * scale)
        s[0].med_sa.append(sa[2] * scale)
        s[0].med_ss.append(ss[0] * scale)
        s[0].med_ss.append(ss[1] * scale)
        s[0].med_ss.append(ss[2] * scale)
        s[0].med_g.append(g_val)
    name_buf.free(); sa.free(); ss.free(); type_buf.free(); val_buf.free(); ia.free()

def lookup_medium(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                  name: UnsafePointer[UInt8, MutAnyOrigin]) -> Int32:
    if name[0] == UInt8(0):
        return Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=name.as_immutable())
    for i in range(len(s[0].med_names)):
        if s[0].med_names[i] == name_str:
            return Int32(i)
    return Int32(-1)

def handle_medium_interface(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                            s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var inside_buf  = alloc[UInt8](64)
    var outside_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, inside_buf, 64)
    _ = scanner_parse_quoted_string(handle, outside_buf, 64)
    s[0].cur_attr.inside_medium  = lookup_medium(s, inside_buf)
    s[0].cur_attr.outside_medium = lookup_medium(s, outside_buf)
    inside_buf.free(); outside_buf.free()

# ── Shape handlers ────────────────────────────────────────────────────────────

def handle_sphere_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                             s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var radius = Float32(1.0)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "radius"):
            radius = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var cx = s[0].ctm[12]
    var cy = s[0].ctm[13]
    var cz = s[0].ctm[14]
    var sx = sqrt(s[0].ctm[0]*s[0].ctm[0] + s[0].ctm[1]*s[0].ctm[1] + s[0].ctm[2]*s[0].ctm[2])
    if sx < Float32(1e-6): sx = Float32(1.0)
    radius *= sx
    s[0].spheres_cx.append(cx)
    s[0].spheres_cy.append(cy)
    s[0].spheres_cz.append(cz)
    s[0].spheres_r.append(radius)
    s[0].spheres_mat.append(s[0].cur_attr.mat_idx)
    s[0].spheres_inside_med.append(s[0].cur_attr.inside_medium)
    s[0].spheres_outside_med.append(s[0].cur_attr.outside_medium)
    s[0].spheres_al.append(s[0].cur_attr.is_alight)
    s[0].spheres_rgb.append(s[0].cur_attr.al_rgb)

def handle_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                     s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var shape_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, shape_type, 64)

    var is_tri = _psc_streq(shape_type, "trianglemesh")
    var is_ply = _psc_streq(shape_type, "plymesh")
    var is_curve = _psc_streq(shape_type, "curve")
    var is_sphere = _psc_streq(shape_type, "sphere")
    shape_type.free()

    if is_curve:
        handle_curve_shape(handle, s)
        return

    if is_sphere:
        handle_sphere_shape(handle, s)
        return

    if not is_tri and not is_ply:
        _psc_skip_params(handle)
        return

    if is_ply:
        var ply_filename = alloc[UInt8](PSC_FILE_MAX)
        ply_filename[0] = UInt8(0)
        var type_buf = alloc[UInt8](64)
        var name_buf = alloc[UInt8](128)
        var ia = alloc[Int32](1)
        ia[0] = Int32(0)
        var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        while found != 0:
            var is_array = ia[0]
            if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
                _ = scanner_parse_quoted_string(handle, ply_filename, PSC_FILE_MAX)
                if is_array:
                    _ = scanner_scan_char(handle, UInt8(93))
            else:
                _psc_skip_value(handle, type_buf, is_array)
                if is_array:
                    _ = scanner_scan_char(handle, UInt8(93))
            ia[0] = Int32(0)
            found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        ia.free()
        type_buf.free(); name_buf.free()

        var full_path = alloc[UInt8](PSC_FILE_MAX * 2)
        var dir_len = s[0].scene_dir.byte_length()
        for ki in range(dir_len):
            full_path[ki] = s[0].scene_dir.unsafe_ptr()[ki]
        var fn_i = 0
        while ply_filename[fn_i] != UInt8(0) and dir_len + fn_i < PSC_FILE_MAX * 2 - 1:
            full_path[dir_len + fn_i] = ply_filename[fn_i]
            fn_i += 1
        full_path[dir_len + fn_i] = UInt8(0)
        ply_filename.free()

        var ply_pts     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_nv      = alloc[Int32](1)
        var ply_idx     = alloc[UnsafePointer[Int32, MutAnyOrigin]](1)
        var ply_nt      = alloc[Int32](1)
        var ply_uvs     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_has_uvs = alloc[Int32](1)
        var ply_nrm     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_has_nrm = alloc[Int32](1)
        ply_uvs[0] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
        ply_has_uvs[0] = Int32(0)
        ply_nrm[0] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
        ply_has_nrm[0] = Int32(0)
        var ok = load_ply(full_path, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
        if ok == 0:
            print("PLY load FAILED:", String(unsafe_from_utf8_ptr=full_path.as_immutable()))
            full_path.free()
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
            return
        full_path.free()
        var nv = ply_nv[0]
        var nt = ply_nt[0]
        if nv <= 0 or nt <= 0:
            ply_pts[0].free(); ply_idx[0].free()
            if ply_has_uvs[0] != 0:
                ply_uvs[0].free()
            if ply_has_nrm[0] != 0:
                ply_nrm[0].free()
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
            return
        var tmp_f2 = ply_pts[0]
        var tmp_i2 = ply_idx[0]
        store_mesh(s, tmp_f2, tmp_i2, nv, nt)
        if ply_has_uvs[0] != 0:
            var uv_ptr = ply_uvs[0]
            var n_uv_floats = Int(nv) * 2
            for uvi in range(n_uv_floats):
                s[0].meshes[len(s[0].meshes) - 1].uvs.append(uv_ptr[uvi])
            uv_ptr.free()
        if ply_has_nrm[0] != 0:
            var nrm_ptr = ply_nrm[0]
            var ctm_inv = alloc[Float32](16)
            _ = matrix_invert(s[0].ctm.unsafe_ptr(), ctm_inv)
            var nrm_world = alloc[Float32](Int(nv) * 3)
            transform_normals(ctm_inv, nrm_ptr, nv, nrm_world)
            ref last_mesh = s[0].meshes[len(s[0].meshes) - 1]
            last_mesh.normals.reserve(Int(nv) * 3)
            for ni in range(Int(nv)):
                var nx = nrm_world[ni*3+0]; var ny = nrm_world[ni*3+1]; var nz = nrm_world[ni*3+2]
                var nlen = sqrt(nx*nx + ny*ny + nz*nz)
                if nlen > Float32(1e-12):
                    var inv = Float32(1.0) / nlen
                    nx *= inv; ny *= inv; nz *= inv
                last_mesh.normals.append(nx)
                last_mesh.normals.append(ny)
                last_mesh.normals.append(nz)
            nrm_world.free(); ctm_inv.free(); nrm_ptr.free()
        tmp_f2.free(); tmp_i2.free()
        ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
        ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
        return

    var tmp_f = alloc[Float32](65536)
    var tmp_i = alloc[Int32](16384)
    var tmp_uv = alloc[Float32](65536)
    var n_pts  = Int32(0)
    var n_idx  = Int32(0)
    var n_uv   = Int32(0)

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        var is_P  = (_psc_streq(name_buf, "P") and _psc_type_is_float(type_buf))
        var is_I  = (_psc_streq(name_buf, "indices") and _psc_type_is_int(type_buf))
        var is_UV = ((_psc_streq(name_buf, "uv") or _psc_streq(name_buf, "st")) and _psc_type_is_float(type_buf))

        if is_P:
            if is_array:
                n_pts = scanner_scan_floats(handle, tmp_f, 65536)
                _ = scanner_scan_char(handle, UInt8(93))
            else:
                _ = scanner_scan_float(handle, tmp_f)
                n_pts = Int32(3)
        elif is_I:
            if is_array:
                n_idx = scanner_scan_ints(handle, tmp_i, 16384)
                _ = scanner_scan_char(handle, UInt8(93))
            else:
                _ = scanner_scan_int(handle, tmp_i)
                n_idx = Int32(1)
        elif is_UV:
            if is_array:
                n_uv = scanner_scan_floats(handle, tmp_uv, 65536)
                _ = scanner_scan_char(handle, UInt8(93))
            else:
                _ = scanner_scan_float(handle, tmp_uv)
                n_uv = Int32(2)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var n_verts = n_pts / Int32(3)
    var n_tris  = n_idx / Int32(3)

    if n_verts <= 0 or n_tris <= 0:
        tmp_f.free(); tmp_i.free(); tmp_uv.free()
        return

    store_mesh(s, tmp_f, tmp_i, n_verts, n_tris)
    if n_uv >= n_verts * Int32(2):
        for ui in range(Int(n_verts) * 2):
            s[0].meshes[len(s[0].meshes) - 1].uvs.append(tmp_uv[ui])
    tmp_f.free(); tmp_i.free(); tmp_uv.free()

# ── Texture handler ───────────────────────────────────────────────────────────

def handle_texture(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                       s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var tex_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, tex_name, PSC_NAME_MAX)
    var tex_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_type, 64)
    var tex_class = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_class, 64)
    if _psc_streq(tex_class, "constant"):
        var ctype = alloc[UInt8](64)
        var cname = alloc[UInt8](128)
        var cia   = alloc[Int32](1); cia[0] = Int32(0)
        var crgb  = alloc[Float32](3)
        crgb[0] = Float32(0.5); crgb[1] = Float32(0.5); crgb[2] = Float32(0.5)
        var cfound = scanner_parse_param_header(handle, ctype, 64, cname, 128, cia)
        while cfound != 0:
            var c_is_array = cia[0]
            if _psc_streq(cname, "value"):
                if ctype[0] == UInt8(102):  # 'f' float -> replicate to all channels
                    var tmp = alloc[Float32](1)
                    _ = scanner_scan_float(handle, tmp)
                    crgb[0] = tmp[0]; crgb[1] = tmp[0]; crgb[2] = tmp[0]
                    tmp.free()
                    if c_is_array:
                        _ = scanner_scan_char(handle, UInt8(93))
                else:
                    _psc_scan_rgb(handle, crgb, c_is_array)
            else:
                _psc_skip_value(handle, ctype, c_is_array)
                if c_is_array:
                    _ = scanner_scan_char(handle, UInt8(93))
            cia[0] = Int32(0)
            cfound = scanner_parse_param_header(handle, ctype, 64, cname, 128, cia)
        s[0].const_tex_names.append(String(unsafe_from_utf8_ptr=tex_name.as_immutable()))
        s[0].const_tex_rgb.append(crgb[0])
        s[0].const_tex_rgb.append(crgb[1])
        s[0].const_tex_rgb.append(crgb[2])
        crgb.free(); ctype.free(); cname.free(); cia.free()
        tex_name.free(); tex_type.free(); tex_class.free()
        return
    if _psc_streq(tex_class, "checkerboard"):
        var c1 = alloc[Float32](3); c1[0]=Float32(0.4); c1[1]=Float32(0.4); c1[2]=Float32(0.4)
        var c2 = alloc[Float32](3); c2[0]=Float32(0.9); c2[1]=Float32(0.9); c2[2]=Float32(0.9)
        var uscale = Float32(1.0)
        var vscale = Float32(1.0)
        var ctype2 = alloc[UInt8](64)
        var cname2 = alloc[UInt8](128)
        var cia2 = alloc[Int32](1); cia2[0] = Int32(0)
        var cfound2 = scanner_parse_param_header(handle, ctype2, 64, cname2, 128, cia2)
        while cfound2 != 0:
            var c_is_arr = cia2[0]
            if _psc_streq(cname2, "tex1") or _psc_streq(cname2, "tex2"):
                var dst = c1 if _psc_streq(cname2, "tex1") else c2
                if ctype2[0] == UInt8(102):  # 'f' float
                    var tmp = _psc_scan_one_float(handle, c_is_arr)
                    dst[0] = tmp; dst[1] = tmp; dst[2] = tmp
                elif _psc_type_is_float(ctype2):
                    _psc_scan_rgb(handle, dst, c_is_arr)
                else:
                    _psc_skip_value(handle, ctype2, c_is_arr)
                    if c_is_arr:
                        _ = scanner_scan_char(handle, UInt8(93))
            elif _psc_streq(cname2, "uscale") and _psc_type_is_float(ctype2):
                uscale = _psc_scan_one_float(handle, c_is_arr)
            elif _psc_streq(cname2, "vscale") and _psc_type_is_float(ctype2):
                vscale = _psc_scan_one_float(handle, c_is_arr)
            else:
                _psc_skip_value(handle, ctype2, c_is_arr)
                if c_is_arr:
                    _ = scanner_scan_char(handle, UInt8(93))
            cia2[0] = Int32(0)
            cfound2 = scanner_parse_param_header(handle, ctype2, 64, cname2, 128, cia2)
        # Bake a 256x256 checker pattern to a temp EXR and register as an imagemap.
        comptime W = 256
        comptime H = 256
        var pixels = alloc[Float32](W * H * 3)
        for j in range(H):
            for i in range(W):
                var cu = Int(Float32(i) / Float32(W) * uscale) & 1
                var cv = Int(Float32(j) / Float32(H) * vscale) & 1
                var src = c1 if (cu ^ cv) == 0 else c2
                var idx = (j * W + i) * 3
                pixels[idx] = src[0]; pixels[idx+1] = src[1]; pixels[idx+2] = src[2]
        var checker_idx = len(s[0].tex_names)
        var fname_str = "/tmp/gonzales_checker_" + String(checker_idx) + ".exr"
        var fname_buf = alloc[UInt8](256)
        var slen = fname_str.byte_length()
        for k in range(slen):
            fname_buf[k] = fname_str.unsafe_ptr()[k]
        fname_buf[slen] = UInt8(0)
        _ = external_call["write_image_rgb", Int32,
            UnsafePointer[UInt8, MutAnyOrigin],
            UnsafePointer[Float32, MutAnyOrigin],
            Int32, Int32, Int32, Int32,
        ](fname_buf, pixels, Int32(W), Int32(H), Int32(W), Int32(H))
        pixels.free(); fname_buf.free()
        c1.free(); c2.free(); ctype2.free(); cname2.free(); cia2.free()
        var name_str2 = String(unsafe_from_utf8_ptr=tex_name.as_immutable())
        s[0].tex_names.append(name_str2)
        s[0].tex_files.append(fname_str)
        tex_name.free(); tex_type.free(); tex_class.free()
        return
    if not _psc_streq(tex_class, "imagemap"):
        tex_name.free(); tex_type.free(); tex_class.free()
        _psc_skip_params(handle)
        return

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            var name_str = String(unsafe_from_utf8_ptr=tex_name.as_immutable())
            var file_str = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            s[0].tex_names.append(name_str)
            s[0].tex_files.append(file_str)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    tex_name.free(); tex_type.free(); tex_class.free()
    type_buf.free(); name_buf.free(); str_val.free()

# ── Main parse loop ───────────────────────────────────────────────────────────

def parse_scene_file(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
              s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var kw_buf = alloc[UInt8](256)
    var ws_delims = alloc[UInt8](4)
    ws_delims[0] = UInt8(32); ws_delims[1] = UInt8(9)
    ws_delims[2] = UInt8(10); ws_delims[3] = UInt8(13)

    while scanner_is_at_end(handle) == 0:
        var n = scanner_scan_token(handle, ws_delims, 4, kw_buf, 256)
        if n < 0:
            break
        if n == 0:
            continue

        if kw_buf[0] == UInt8(35):  # '#'
            _psc_skip_line(handle)
            continue

        if _psc_streq(kw_buf, "Integrator"):
            _psc_handle_integrator(handle, s)
        elif _psc_streq(kw_buf, "Sampler"):
            _psc_handle_sampler(handle, s)
        elif _psc_streq(kw_buf, "PixelFilter"):
            _psc_handle_filter(handle, s)
        elif _psc_streq(kw_buf, "Film"):
            _psc_handle_film(handle, s)
        elif _psc_streq(kw_buf, "Camera"):
            _psc_handle_camera(handle, s)
        elif _psc_streq(kw_buf, "Transform"):
            _psc_handle_transform(handle, s)
        elif _psc_streq(kw_buf, "Translate"):
            _psc_handle_translate(handle, s)
        elif _psc_streq(kw_buf, "Scale"):
            _psc_handle_scale_kw(handle, s)
        elif _psc_streq(kw_buf, "Rotate"):
            _psc_handle_rotate(handle, s)
        elif _psc_streq(kw_buf, "LookAt"):
            _psc_handle_lookat(handle, s)
        elif _psc_streq(kw_buf, "WorldBegin"):
            _psc_handle_world_begin(s)
        elif _psc_streq(kw_buf, "WorldEnd"):
            break
        elif _psc_streq(kw_buf, "MakeNamedMaterial"):
            _psc_handle_make_named_material(handle, s)
        elif _psc_streq(kw_buf, "NamedMaterial"):
            _psc_handle_named_material(handle, s)
        elif _psc_streq(kw_buf, "ObjectBegin"):
            var obj_name = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            obj_name.free()
            s[0].object_depth += 1
        elif _psc_streq(kw_buf, "ObjectEnd"):
            if s[0].object_depth > 0:
                s[0].object_depth -= 1
        elif _psc_streq(kw_buf, "ObjectInstance"):
            var obj_name = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            obj_name.free()
        elif _psc_streq(kw_buf, "Shape"):
            if s[0].object_depth == 0:
                handle_shape(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "AttributeBegin"):
            _psc_handle_attribute_begin(s)
        elif _psc_streq(kw_buf, "AttributeEnd"):
            _psc_handle_attribute_end(s)
        elif _psc_streq(kw_buf, "TransformBegin"):
            ctm_push(s[0])
        elif _psc_streq(kw_buf, "TransformEnd"):
            ctm_pop(s[0])
        elif _psc_streq(kw_buf, "ReverseOrientation"):
            s[0].cur_attr.reverse_orient = not s[0].cur_attr.reverse_orient
        elif _psc_streq(kw_buf, "AreaLightSource"):
            if s[0].object_depth == 0:
                _psc_handle_area_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "LightSource"):
            if s[0].object_depth == 0:
                handle_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "Texture"):
            handle_texture(handle, s)
        elif _psc_streq(kw_buf, "Include") or _psc_streq(kw_buf, "Import"):
            var inc_name = alloc[UInt8](PSC_FILE_MAX)
            _ = scanner_parse_quoted_string(handle, inc_name, PSC_FILE_MAX)
            var inc_path = alloc[UInt8](PSC_FILE_MAX * 2)
            var dlen = s[0].scene_dir.byte_length()
            for ki in range(dlen):
                inc_path[ki] = s[0].scene_dir.unsafe_ptr()[ki]
            var fi = 0
            while inc_name[fi] != UInt8(0) and dlen + fi < PSC_FILE_MAX * 2 - 1:
                inc_path[dlen + fi] = inc_name[fi]
                fi += 1
            inc_path[dlen + fi] = UInt8(0)
            var sub_handle = scanner_open(inc_path)
            if scanner_is_at_end(sub_handle) == 0:
                parse_scene_file(sub_handle, s)
            else:
                var inc_str = String(unsafe_from_utf8_ptr=inc_name.as_immutable())
                if inc_str.endswith(".xz"):
                    print("Warning: cannot open include (decompress first with xz -dk):", inc_str)
                else:
                    print("Warning: cannot open include:", inc_str)
            scanner_free(sub_handle)
            inc_name.free(); inc_path.free()
        elif _psc_streq(kw_buf, "Material"):
            _psc_handle_make_named_material(handle, s)
            s[0].cur_attr.mat_idx = Int32(len(s[0].named_materials)) - Int32(1)
        elif _psc_streq(kw_buf, "MakeNamedMedium"):
            handle_named_medium(handle, s)
        elif _psc_streq(kw_buf, "MediumInterface"):
            handle_medium_interface(handle, s)
        elif _psc_streq(kw_buf, "ConcatTransform"):
            _ = scanner_scan_char(handle, UInt8(91))  # '['
            var tmp = alloc[Float32](16)
            for i in range(16):
                _ = scanner_scan_float(handle, tmp + i)
            _ = scanner_scan_char(handle, UInt8(93))  # ']'
            var result = alloc[Float32](16)
            matrix_multiply(s[0].ctm.unsafe_ptr(), tmp, result)
            for i in range(16):
                s[0].ctm[i] = result[i]
            tmp.free(); result.free()
        else:
            _ = scanner_parse_quoted_string(handle, kw_buf, 256)
            _psc_skip_params(handle)

    kw_buf.free()
    ws_delims.free()

# ── Camera/film matrix helpers ────────────────────────────────────────────────

def make_perspective_matrix(fov_deg: Float32, near: Float32,
                         dst: UnsafePointer[Float32, MutAnyOrigin]):
    var half_rad = fov_deg * PI / Float32(360)
    var inv_tan = Float32(1) / tan(half_rad)
    var far = fov_deg
    var t22 = far / (far - near)
    var t23 = -(far * near) / (far - near)
    for i in range(16):
        dst[i] = Float32(0)
    dst[0]  = inv_tan
    dst[5]  = inv_tan
    dst[10] = t22
    dst[11] = Float32(1)
    dst[14] = t23

def make_screen_to_raster(fw: Int32, fh: Int32,
                               smin_x: Float32, smax_x: Float32,
                               smin_y: Float32, smax_y: Float32,
                               dst: UnsafePointer[Float32, MutAnyOrigin]):
    var sx = Float32(fw) / (smax_x - smin_x)
    var sy = Float32(fh) / (smin_y - smax_y)
    var tx = -smin_x * sx
    var ty = -smax_y * sy
    for i in range(16):
        dst[i] = Float32(0)
    dst[0]  = sx
    dst[5]  = sy
    dst[10] = Float32(1)
    dst[15] = Float32(1)
    dst[12] = tx
    dst[13] = ty

# ── Scene finalization ────────────────────────────────────────────────────────

def finalize_scene(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                 psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
                 verbose: Bool = False):

    # ---- Camera matrices ----
    var c2w = alloc[Float32](16)
    var cam2w_tmp = alloc[Float32](16)
    for i in range(16): cam2w_tmp[i] = s[0].cam2w_raw[i]
    _ = matrix_invert(cam2w_tmp, c2w)
    cam2w_tmp.free()
    psc[0].camera_to_world = c2w

    if verbose:
        print("=== GONZALES DEBUG: Scene Summary ===")
        print("  Camera position (c2w col3):", c2w[12], c2w[13], c2w[14])
        print("  Camera forward (-Z):", -c2w[8], -c2w[9], -c2w[10])
        print("  Camera FOV:", s[0].camera_fov)
        print("  Film:", s[0].film_w, "x", s[0].film_h)
        print("  Meshes:", len(s[0].meshes))
        print("  Named materials:", len(s[0].named_materials))
        print("=== END DEBUG ===")

    var cts = alloc[Float32](16)
    make_perspective_matrix(s[0].camera_fov, Float32(0.01), cts)

    var frame = Float32(s[0].film_w) / Float32(s[0].film_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = alloc[Float32](16)
    make_screen_to_raster(s[0].film_w, s[0].film_h,
                                smin_x, smax_x, smin_y, smax_y, str_mat)

    var rts = alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)

    var cts_inv = alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)

    var r2c = alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c)
    psc[0].raster_to_camera = r2c

    cts.free(); str_mat.free(); rts.free(); cts_inv.free()

    # ---- Materials ----
    var n_regular = len(s[0].named_materials)

    var n_al = 0
    for i in range(len(s[0].meshes)):
        if s[0].meshes[i].is_area_light:
            n_al += 1

    var n_mats = n_regular + n_al
    var mats = alloc[Material_C](max(n_mats, 1))
    for i in range(n_regular):
        var nm3 = s[0].named_materials[i]
        var mt = nm3.kind
        var ior = nm3.ior
        mats[i].type = mt
        mats[i].tex_idx = nm3.tex_idx
        mats[i].roughU  = nm3.roughness_u
        mats[i].roughV  = nm3.roughness_v
        mats[i].normal_tex_idx = nm3.normal_tex_idx
        mats[i].medium_interface_idx = Int32(-1)
        if mt == Int8(4):
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(5):
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(7):
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(9):
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(8):
            var idx1 = Int32(0)
            var idx2 = Int32(0)
            for j in range(n_regular):
                if s[0].named_materials[j].name == nm3.mix_name1: idx1 = Int32(j)
                if s[0].named_materials[j].name == nm3.mix_name2: idx2 = Int32(j)
            mats[i].tex_idx = (idx2 << 16) | (idx1 & Int32(0xFFFF))
            mats[i].roughU  = nm3.mix_amount
            mats[i].albedo  = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(6):
            mats[i].albedo = nm3.albedo
            mats[i].emission = nm3.transmittance
        elif mt == Int8(11):
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
            mats[i].type = Int8(1)
        else:
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))

    # ---- Meshes + area lights ----
    var n_meshes = len(s[0].meshes)
    var meshes   = alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts  = alloc[UnsafePointer[Float32, MutAnyOrigin]](max(n_meshes, 1))
    var out_vis  = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_fis  = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_nv    = alloc[Int32](max(n_meshes, 1))
    var out_nt    = alloc[Int32](max(n_meshes, 1))
    var out_uv_nv = alloc[Int32](max(n_meshes, 1))
    var out_nrm_nv = alloc[Int32](max(n_meshes, 1))

    var al_list  = alloc[AreaLight_C](max(n_al, 1))
    var al_count = Int32(0)
    var al_mat_base = n_regular

    for i in range(n_meshes):
        ref ma = s[0].meshes[i]
        var nv = len(ma.points) // 4
        var nt = len(ma.face_idxs)
        var pts_c = alloc[Float32](nv * 4)
        for vi in range(nv * 4): pts_c[vi] = ma.points[vi]
        var vis_c = alloc[Int64](nt * 3)
        for ti2 in range(nt * 3): vis_c[ti2] = ma.vert_idxs[ti2]
        var fis_c = alloc[Int64](nt)
        for ti2 in range(nt): fis_c[ti2] = ma.face_idxs[ti2]
        out_pts[i] = pts_c
        out_vis[i] = vis_c
        out_fis[i] = fis_c
        out_nv[i]  = Int32(nv)
        out_nt[i]  = Int32(nt)
        meshes[i].points        = pts_c
        meshes[i].vertexIndices = vis_c
        meshes[i].faceIndices   = fis_c
        if len(ma.uvs) >= nv * 2:
            var uv_c = alloc[Float32](nv * 2)
            for ui in range(nv * 2): uv_c[ui] = ma.uvs[ui]
            meshes[i].uvs = uv_c
            out_uv_nv[i] = Int32(nv)
        else:
            meshes[i].uvs = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            out_uv_nv[i] = Int32(0)
        if len(ma.normals) >= nv * 3:
            var nrm_c = alloc[Float32](nv * 3)
            for ni in range(nv * 3): nrm_c[ni] = ma.normals[ni]
            meshes[i].normals = nrm_c
            out_nrm_nv[i] = Int32(nv)
        else:
            meshes[i].normals = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            out_nrm_nv[i] = Int32(0)

        if ma.is_area_light:
            var al_idx = Int(al_count)
            var em = ma.al_rgb
            var t_area  = Float32(0.0)
            for ti in range(nt):
                var vi0 = Int(vis_c[ti*3+0]) * 4
                var vi1 = Int(vis_c[ti*3+1]) * 4
                var vi2 = Int(vis_c[ti*3+2]) * 4
                var ex = pts_c[vi1+0] - pts_c[vi0+0]
                var ey = pts_c[vi1+1] - pts_c[vi0+1]
                var ez = pts_c[vi1+2] - pts_c[vi0+2]
                var fx = pts_c[vi2+0] - pts_c[vi0+0]
                var fy = pts_c[vi2+1] - pts_c[vi0+1]
                var fz = pts_c[vi2+2] - pts_c[vi0+2]
                var cxv = ey*fz - ez*fy
                var cyv = ez*fx - ex*fz
                var czv = ex*fy - ey*fx
                t_area += Float32(0.5) * sqrt(cxv*cxv + cyv*cyv + czv*czv)
            al_list[al_idx].meshIdx    = Int32(i)
            al_list[al_idx].n_tris     = Int32(nt)
            al_list[al_idx].emission   = em
            al_list[al_idx].total_area = t_area
            mats[al_mat_base + al_idx].type     = Int8(2)
            mats[al_mat_base + al_idx].albedo   = RGB(Float32(0), Float32(0), Float32(0))
            mats[al_mat_base + al_idx].emission = em
            mats[al_mat_base + al_idx].tex_idx  = Int32(-1)
            mats[al_mat_base + al_idx].roughU   = Float32(0)
            mats[al_mat_base + al_idx].roughV   = Float32(0)
            mats[al_mat_base + al_idx].normal_tex_idx = Int32(-1)
            mats[al_mat_base + al_idx].medium_interface_idx = Int32(-1)
            al_count += 1

    # ---- BVH construction ----
    var n_with_mi = 0
    for mi in range(n_meshes):
        if s[0].meshes[mi].inside_medium >= Int32(0) or s[0].meshes[mi].outside_medium >= Int32(0):
            n_with_mi += 1
    for si in range(len(s[0].spheres_cx)):
        if s[0].spheres_inside_med[si] >= Int32(0) or s[0].spheres_outside_med[si] >= Int32(0):
            n_with_mi += 1

    if n_with_mi > 0:
        var expanded_n = n_mats + n_with_mi
        var new_mats = alloc[Material_C](expanded_n)
        for ci in range(n_mats):
            new_mats[ci] = mats[ci]
        mats.free()
        mats = new_mats

        var iface_buf = alloc[MediumInterface_C](n_with_mi)
        var dup_idx = n_mats
        var iface_idx = 0

        for mi in range(n_meshes):
            var ins = s[0].meshes[mi].inside_medium
            var out = s[0].meshes[mi].outside_medium
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[0].meshes[mi].mat_idx)
            if orig_mat < 0: continue
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            s[0].meshes[mi].mat_idx = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        for si in range(len(s[0].spheres_cx)):
            var ins = s[0].spheres_inside_med[si]
            var out = s[0].spheres_outside_med[si]
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[0].spheres_mat[si])
            if orig_mat < 0: continue
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            s[0].spheres_mat[si] = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        n_mats = dup_idx
        psc[0].medium_ifaces = iface_buf
        psc[0].medium_iface_count = Int32(iface_idx)
    else:
        psc[0].medium_ifaces = UnsafePointer[MediumInterface_C, MutAnyOrigin].unsafe_dangling()
        psc[0].medium_iface_count = Int32(0)

    var total_tris = Int32(0)
    for i in range(n_meshes):
        total_tris += Int32(len(s[0].meshes[i].face_idxs))

    var prim_bounds = alloc[Float32](Int(total_tris) * 6)
    var tri_mesh    = alloc[Int32](Int(total_tris))
    var tri_local   = alloc[Int32](Int(total_tris))

    var flat_idx = Int32(0)
    for mi in range(n_meshes):
        var pts = out_pts[mi]
        var vis = out_vis[mi]
        var nt  = Int(out_nt[mi])
        for ti in range(nt):
            var v0 = Int(vis[ti*3+0]) * 4
            var v1 = Int(vis[ti*3+1]) * 4
            var v2 = Int(vis[ti*3+2]) * 4
            var x0 = pts[v0]; var y0 = pts[v0+1]; var z0 = pts[v0+2]
            var x1 = pts[v1]; var y1 = pts[v1+1]; var z1 = pts[v1+2]
            var x2 = pts[v2]; var y2 = pts[v2+1]; var z2 = pts[v2+2]
            var b = Int(flat_idx) * 6
            prim_bounds[b+0] = min(x0, min(x1, x2))
            prim_bounds[b+1] = min(y0, min(y1, y2))
            prim_bounds[b+2] = min(z0, min(z1, z2))
            prim_bounds[b+3] = max(x0, max(x1, x2))
            prim_bounds[b+4] = max(y0, max(y1, y2))
            prim_bounds[b+5] = max(z0, max(z1, z2))
            tri_mesh[Int(flat_idx)]  = Int32(mi)
            tri_local[Int(flat_idx)] = Int32(ti)
            flat_idx += 1

    var max_bvh_nodes = Int(total_tris) * 2 + 4
    var bvh_nodes = alloc[BVH2Node](max_bvh_nodes)
    var bvh_order = alloc[Int32](Int(total_tris))
    var node_count = build_bvh2(prim_bounds, total_tris, bvh_nodes, bvh_order)

    prim_bounds.free()

    var prim_ids = alloc[PrimId_C](Int(total_tris))

    var mesh_al_idx = alloc[Int32](max(n_meshes, 1))
    var running_al = Int32(0)
    for mi in range(n_meshes):
        if s[0].meshes[mi].is_area_light:
            mesh_al_idx[mi] = running_al
            running_al += 1
        else:
            mesh_al_idx[mi] = Int32(-1)

    for k in range(Int(total_tris)):
        var orig = Int(bvh_order[k])
        var mi   = Int(tri_mesh[orig])
        var ti   = Int(tri_local[orig])
        if s[0].meshes[mi].is_area_light:
            var al_idx = Int(mesh_al_idx[mi])
            prim_ids[k].type          = Int8(3)
            prim_ids[k].id1           = Int64(al_idx)
            prim_ids[k].id2           = (Int64(mi) << 32) | Int64(ti)
            prim_ids[k].materialIndex = Int64(al_mat_base + al_idx)
        else:
            var mat_idx = Int(s[0].meshes[mi].mat_idx)
            prim_ids[k].type          = Int8(0)
            prim_ids[k].id1           = Int64(mi)
            prim_ids[k].id2           = Int64(ti * 3)
            prim_ids[k].materialIndex = Int64(max(mat_idx, 0))
        prim_ids[k]._pad0 = Int8(0); prim_ids[k]._pad1 = Int8(0)
        prim_ids[k]._pad2 = Int8(0); prim_ids[k]._pad3 = Int8(0)
        prim_ids[k]._pad4 = Int8(0); prim_ids[k]._pad5 = Int8(0)
        prim_ids[k]._pad6 = Int8(0)

    tri_mesh.free(); tri_local.free(); bvh_order.free(); mesh_al_idx.free()

    # ---- Sampler params ----
    var spp = s[0].samples_per_pixel
    var log2_spp = Int32(0)
    var tmp_spp = spp
    while tmp_spp > Int32(1):
        tmp_spp >>= 1
        log2_spp += 1
    var log4_spp = (log2_spp + Int32(1)) / Int32(2)
    var dim = max(s[0].film_w, s[0].film_h)
    var log2_dim = Int32(0)
    var tmp_dim = dim
    while tmp_dim > Int32(1):
        tmp_dim >>= 1
        log2_dim += 1
    var n_base4 = log2_dim + log4_spp

    # ---- Filter norms ----
    var norm_x = gaussian_norm(s[0].filter_support_x, s[0].filter_sigma)
    var norm_y = gaussian_norm(s[0].filter_support_y, s[0].filter_sigma)
    var fweight = Float32(1.0)

    # ---- RNG seed from time ----
    var rng_seed = UInt64(perf_counter_ns())

    # ---- Film filename copy ----
    var fname = alloc[UInt8](PSC_FILE_MAX)
    var fnstr = s[0].film_filename
    var fnlen = min(fnstr.byte_length(), PSC_FILE_MAX - 1)
    for fi in range(fnlen): fname[fi] = fnstr.unsafe_ptr()[fi]
    fname[fnlen] = UInt8(0)

    # ---- Texture filename table ----
    var n_tex = len(s[0].tex_names)
    var tex_ptrs = alloc[UnsafePointer[UInt8, MutAnyOrigin]](max(n_tex, 1))
    for ti in range(n_tex):
        var fstr = s[0].tex_files[ti]
        var slen = fstr.byte_length()
        var copy = alloc[UInt8](slen + 1)
        for ci in range(slen): copy[ci] = fstr.unsafe_ptr()[ci]
        copy[slen] = UInt8(0)
        tex_ptrs[ti] = copy

    # ---- Fill output struct ----
    psc[0].materials        = mats
    psc[0].material_count   = Int32(n_mats)
    psc[0].area_lights      = al_list
    psc[0].area_light_count = al_count
    psc[0].meshes           = meshes
    psc[0].mesh_pts         = out_pts
    psc[0].mesh_vis         = out_vis
    psc[0].mesh_fis         = out_fis
    psc[0].mesh_n_verts     = out_nv
    psc[0].mesh_n_tris      = out_nt
    psc[0].mesh_uv_n_verts  = out_uv_nv
    psc[0].mesh_nrm_n_verts = out_nrm_nv
    psc[0].mesh_count       = Int32(n_meshes)
    psc[0].bvh_nodes        = bvh_nodes
    psc[0].prim_ids         = prim_ids
    psc[0].bvh_node_count   = node_count
    psc[0].prim_count       = total_tris
    psc[0].film_w           = s[0].film_w
    psc[0].film_h           = s[0].film_h
    psc[0].camera_fov       = s[0].camera_fov
    psc[0].film_iso         = s[0].film_iso
    psc[0].film_max_comp    = s[0].film_max_comp
    psc[0].film_filename    = fname
    psc[0].filter_sigma     = s[0].filter_sigma
    psc[0].filter_support_x = s[0].filter_support_x
    psc[0].filter_support_y = s[0].filter_support_y
    psc[0].filter_type      = s[0].filter_type
    psc[0].filter_norm_x    = norm_x
    psc[0].filter_norm_y    = norm_y
    psc[0].filter_weight    = fweight
    psc[0].camera_fov       = s[0].camera_fov
    psc[0].samples_per_pixel = spp
    psc[0].log2_spp         = log2_spp
    psc[0].n_base4_digits   = n_base4
    psc[0].max_depth        = s[0].max_depth
    psc[0].rng_seed         = rng_seed
    psc[0].tex_filenames    = tex_ptrs
    psc[0].tex_count        = Int32(n_tex)

    # ---- Non-area lights ----
    var nd = len(s[0].distant_dirs) // 3
    if nd > 0:
        var dl_buf = alloc[DistantLight_C](nd)
        for i in range(nd):
            dl_buf[i] = DistantLight_C(
                Vec3f(s[0].distant_dirs[i*3+0], s[0].distant_dirs[i*3+1], s[0].distant_dirs[i*3+2]),
                Float32(0),
                RGB(s[0].distant_rgbs[i*3+0], s[0].distant_rgbs[i*3+1], s[0].distant_rgbs[i*3+2]),
                Float32(0))
        psc[0].distant_lights = dl_buf
    else:
        psc[0].distant_lights = UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].distant_count = Int32(nd)

    var np2 = len(s[0].point_pos) // 3
    if np2 > 0:
        var pl_buf = alloc[PointLight_C](np2)
        for i in range(np2):
            pl_buf[i] = PointLight_C(
                Point3f(s[0].point_pos[i*3+0], s[0].point_pos[i*3+1], s[0].point_pos[i*3+2]),
                Float32(0),
                RGB(s[0].point_rgbs[i*3+0], s[0].point_rgbs[i*3+1], s[0].point_rgbs[i*3+2]),
                Float32(0))
        psc[0].point_lights = pl_buf
    else:
        psc[0].point_lights = UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].point_count = Int32(np2)

    var ni = len(s[0].inf_tex_idx)
    if ni > 0:
        var il_buf = alloc[InfiniteLight_C](ni)
        for i in range(ni):
            var tidx = s[0].inf_tex_idx[i]
            var sc = RGB(s[0].inf_rgb[i*3+0], s[0].inf_rgb[i*3+1], s[0].inf_rgb[i*3+2])
            var cdf_w = Int32(0); var cdf_h = Int32(0)
            var cdf_ptr = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            var raw_pixels = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            if tidx >= Int32(0):
                var fname2 = psc[0].tex_filenames[Int(tidx)]
                var pixels_ptr = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
                var iw_out = alloc[Int32](1); var ih_out = alloc[Int32](1)
                iw_out[0] = Int32(0); ih_out[0] = Int32(0)
                var load_ok = external_call["load_texture_rgb", Int32,
                    UnsafePointer[UInt8, MutAnyOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin], UnsafePointer[Int32, MutAnyOrigin],
                    Int32](
                    fname2, pixels_ptr, iw_out, ih_out, Int32(0))
                var iw = Int(iw_out[0]); var ih = Int(ih_out[0])
                iw_out.free(); ih_out.free()
                if load_ok != Int32(0) and iw > 0 and ih > 0:
                    var pixels = pixels_ptr[0]
                    var tmp_row = alloc[Float32](iw * 3)
                    for r in range(ih // 2):
                        var r2 = ih - 1 - r
                        var off1 = r * iw * 3
                        var off2 = r2 * iw * 3
                        for c in range(iw * 3):
                            tmp_row[c] = pixels[off1 + c]
                            pixels[off1 + c] = pixels[off2 + c]
                            pixels[off2 + c] = tmp_row[c]
                    tmp_row.free()
                    raw_pixels = pixels
                    var cdf_size = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = alloc[Float32](cdf_size)
                    var row_sums = alloc[Float32](ih)
                    for ry in range(ih):
                        var row_sum = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            row_sum += lum
                        row_sums[ry] = row_sum
                    cdf_buf[0] = Float32(0.0)
                    for ry in range(ih):
                        cdf_buf[ry + 1] = cdf_buf[ry] + row_sums[ry]
                    var total = cdf_buf[ih]
                    if total > Float32(0.0):
                        var inv_total = Float32(1.0) / total
                        for ry in range(ih + 1):
                            cdf_buf[ry] *= inv_total
                    for ry in range(ih):
                        var base = (ih + 1) + ry * (iw + 1)
                        cdf_buf[base] = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            cdf_buf[base + rx + 1] = cdf_buf[base + rx] + lum
                        var row_total = cdf_buf[base + iw]
                        if row_total > Float32(0.0):
                            var inv_rt = Float32(1.0) / row_total
                            for rx in range(iw + 1):
                                cdf_buf[base + rx] *= inv_rt
                    row_sums.free()
                    cdf_ptr = cdf_buf
                    cdf_w = Int32(iw); cdf_h = Int32(ih)
                pixels_ptr.free()
            var w2l = alloc[Float32](16)
            var light_ctm_base = i * 16
            var is_identity = True
            for ci in range(16):
                var expected = Float32(1) if (ci == 0 or ci == 5 or ci == 10 or ci == 15) else Float32(0)
                if s[0].inf_ctm[light_ctm_base + ci] != expected:
                    is_identity = False
                    break
            if is_identity:
                _psc_identity(w2l)
            else:
                var light_ctm_tmp = alloc[Float32](16)
                for ci in range(16): light_ctm_tmp[ci] = s[0].inf_ctm[light_ctm_base + ci]
                _ = matrix_invert(light_ctm_tmp, w2l)
                light_ctm_tmp.free()
            il_buf[i] = InfiniteLight_C(sc, tidx, cdf_w, cdf_h, cdf_ptr, raw_pixels, w2l)
        psc[0].infinite_lights = il_buf
    else:
        psc[0].infinite_lights = UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].infinite_count = Int32(ni)

    # ---- Analytical spheres ----
    var ns = len(s[0].spheres_cx)
    if ns > 0:
        var sph_buf = alloc[Sphere_C](ns)
        for i in range(ns):
            var em = SampledSpectrum(s[0].spheres_rgb[i].r, s[0].spheres_rgb[i].g, s[0].spheres_rgb[i].b)
            var al_flag = Int8(1) if s[0].spheres_al[i] else Int8(0)
            sph_buf[i] = Sphere_C(
                Point3f(s[0].spheres_cx[i], s[0].spheres_cy[i], s[0].spheres_cz[i]),
                s[0].spheres_r[i],
                s[0].spheres_mat[i],
                al_flag,
                Int8(0), Int8(0), Int8(0),
                em)
        psc[0].spheres = sph_buf
    else:
        psc[0].spheres = UnsafePointer[Sphere_C, MutAnyOrigin].unsafe_dangling()
    psc[0].sphere_count = Int32(ns)

    # ---- Media ----
    var nm = len(s[0].med_g)
    if nm > 0:
        var med_buf = alloc[Medium_C](nm)
        for i in range(nm):
            var sa = SampledSpectrum(s[0].med_sa[i*3], s[0].med_sa[i*3+1], s[0].med_sa[i*3+2])
            var ss = SampledSpectrum(s[0].med_ss[i*3], s[0].med_ss[i*3+1], s[0].med_ss[i*3+2])
            med_buf[i] = Medium_C(sa, ss, s[0].med_g[i],
                                  Float32(0), Float32(0), Float32(0))
        psc[0].mediums = med_buf
    else:
        psc[0].mediums = UnsafePointer[Medium_C, MutAnyOrigin].unsafe_dangling()
    psc[0].medium_count = Int32(nm)

    # ---- Build power-weighted area light CDF ----
    var ls_n = Int(psc[0].area_light_count)
    var ls_cdf = alloc[Float32](max(ls_n + 1, 2))
    ls_cdf[0] = Float32(0.0)
    var ls_total_power = Float32(0.0)
    for i in range(ls_n):
        var al = psc[0].area_lights[i]
        var power = al.emission.luma() * al.total_area
        ls_total_power += power
        ls_cdf[i + 1] = ls_cdf[i] + power
    if ls_total_power > Float32(0.0):
        var inv = Float32(1.0) / ls_total_power
        for i in range(1, ls_n + 1):
            ls_cdf[i] *= inv
    else:
        for i in range(1, ls_n + 1):
            ls_cdf[i] = Float32(i) / Float32(max(ls_n, 1))
    psc[0].light_sampler = LightSampler_C(ls_cdf, Int32(ls_n), Int32(0))

# ── Exported API ──────────────────────────────────────────────────────────────

def resize_film(psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
               new_w: Int32, new_h: Int32):
    psc[0].film_w = new_w
    psc[0].film_h = new_h

    var frame = Float32(new_w) / Float32(new_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = alloc[Float32](16)
    make_screen_to_raster(new_w, new_h, smin_x, smax_x, smin_y, smax_y, str_mat)
    var rts = alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)
    var cts = alloc[Float32](16)
    make_perspective_matrix(psc[0].camera_fov, Float32(0.01), cts)
    var cts_inv = alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)
    if Int(psc[0].raster_to_camera) > 1:
        psc[0].raster_to_camera.free()
    var r2c = alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c)
    psc[0].raster_to_camera = r2c
    cts.free(); str_mat.free(); rts.free(); cts_inv.free()

def mojo_parse_scene(path: UnsafePointer[UInt8, MutAnyOrigin],
                     verbose: Bool = False,
                    ) -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    external_call["createTextureSystem", NoneType]()
    var handle = scanner_open(path)
    if handle[0].is_at_end != Int32(0):
        print("Error: cannot open scene file:", String(unsafe_from_utf8_ptr=path.as_immutable()))
        scanner_free(handle)
        return UnsafePointer[ParsedScene_Mojo, MutAnyOrigin].unsafe_dangling()

    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    var pi = 0
    while path[pi] != UInt8(0):
        pi += 1
    var last_slash = -1
    for ki in range(pi):
        if path[ki] == UInt8(47):
            last_slash = ki
    if last_slash >= 0:
        var dir_tmp = alloc[UInt8](last_slash + 2)
        for ki in range(last_slash + 1):
            dir_tmp[ki] = path[ki]
        dir_tmp[last_slash + 1] = UInt8(0)
        s_ptr[0].scene_dir = String(unsafe_from_utf8_ptr=dir_tmp.as_immutable())
        dir_tmp.free()
    parse_scene_file(handle, s_ptr)
    scanner_free(handle)
    flush_hair(s_ptr)

    var psc = alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, verbose)
    _ = s_ptr.take_pointee()
    s_ptr.free()
    return psc

def mojo_parsed_free(psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]):
    if Int(psc) == 0:
        return
    var n = Int(psc[0].mesh_count)
    for i in range(n):
        psc[0].mesh_pts[i].free()
        psc[0].mesh_vis[i].free()
        psc[0].mesh_fis[i].free()
        if psc[0].mesh_uv_n_verts[i] > Int32(0):
            psc[0].meshes[i].uvs.free()
        if psc[0].mesh_nrm_n_verts[i] > Int32(0):
            psc[0].meshes[i].normals.free()
    if psc[0].mesh_count > 0:
        psc[0].mesh_pts.free()
        psc[0].mesh_vis.free()
        psc[0].mesh_fis.free()
        psc[0].mesh_n_verts.free()
        psc[0].mesh_n_tris.free()
        psc[0].mesh_uv_n_verts.free()
        psc[0].mesh_nrm_n_verts.free()
        psc[0].meshes.free()
    if psc[0].material_count > 0:
        psc[0].materials.free()
    if psc[0].area_light_count > 0:
        psc[0].area_lights.free()
    if psc[0].bvh_node_count > 0:
        psc[0].bvh_nodes.free()
    if psc[0].prim_count > 0:
        psc[0].prim_ids.free()
    if Int(psc[0].raster_to_camera) > 4:
        psc[0].raster_to_camera.free()
    if Int(psc[0].camera_to_world) > 4:
        psc[0].camera_to_world.free()
    if Int(psc[0].film_filename) > 1:
        psc[0].film_filename.free()
    if psc[0].tex_count > 0:
        var nt = Int(psc[0].tex_count)
        for ti in range(nt):
            psc[0].tex_filenames[ti].free()
        psc[0].tex_filenames.free()
    if psc[0].distant_count > 0:
        psc[0].distant_lights.free()
    if psc[0].point_count > 0:
        psc[0].point_lights.free()
    if psc[0].infinite_count > 0:
        var ni = Int(psc[0].infinite_count)
        for ii in range(ni):
            var il = psc[0].infinite_lights[ii]
            if Int(il.cdf_ptr) > 4:
                il.cdf_ptr.free()
            if Int(il.pixels_ptr) > 4:
                _ = external_call["free_texture_rgb", Int32,
                    UnsafePointer[Float32, MutAnyOrigin]](il.pixels_ptr)
            il.world_to_light.free()
        psc[0].infinite_lights.free()
    if psc[0].sphere_count > 0:
        psc[0].spheres.free()
    if Int(psc[0].light_sampler.cdf) > 4:
        psc[0].light_sampler.cdf.free()
    psc.free()

def mojo_apply_overrides(
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    spp_override: Int32,
    w_override: Int32,
    h_override: Int32,
):
    if spp_override > Int32(0):
        var spp = spp_override
        var log2_spp = Int32(0)
        var tmp = spp
        while tmp > Int32(1):
            tmp >>= 1
            log2_spp += 1
        var log4_spp = (log2_spp + Int32(1)) / Int32(2)
        var dim = max(psc[0].film_w, psc[0].film_h)
        var log2_dim = Int32(0)
        var tmp_dim = dim
        while tmp_dim > Int32(1):
            tmp_dim >>= 1
            log2_dim += 1
        psc[0].samples_per_pixel = spp
        psc[0].log2_spp          = log2_spp
        psc[0].n_base4_digits    = log2_dim + log4_spp

    if w_override > Int32(0) and h_override > Int32(0):
        psc[0].film_w = w_override
        psc[0].film_h = h_override
        var cts = alloc[Float32](16)
        make_perspective_matrix(psc[0].camera_fov, Float32(0.01), cts)
        var frame = Float32(w_override) / Float32(h_override)
        var smin_x: Float32; var smax_x: Float32
        var smin_y: Float32; var smax_y: Float32
        if frame >= Float32(1):
            smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
        else:
            smin_x = Float32(-1); smax_x = Float32(1)
            smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame
        var str_mat = alloc[Float32](16)
        make_screen_to_raster(w_override, h_override,
                              smin_x, smax_x, smin_y, smax_y, str_mat)
        var rts = alloc[Float32](16)
        _ = matrix_invert(str_mat, rts)
        var cts_inv = alloc[Float32](16)
        _ = matrix_invert(cts, cts_inv)
        matrix_multiply(cts_inv, rts, psc[0].raster_to_camera)
        cts.free(); str_mat.free(); rts.free(); cts_inv.free()

        var log2_spp = psc[0].log2_spp
        var log4_spp = (log2_spp + Int32(1)) / Int32(2)
        var dim = max(w_override, h_override)
        var log2_dim = Int32(0)
        var tmp_dim = dim
        while tmp_dim > Int32(1):
            tmp_dim >>= 1
            log2_dim += 1
        psc[0].n_base4_digits = log2_dim + log4_spp

def mojo_parsed_scene_descriptor(
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]
) -> UnsafePointer[SceneDescriptor2_C, MutAnyOrigin]:
    var sd = alloc[SceneDescriptor2_C](1)
    sd[0].bvh2Nodes        = psc[0].bvh_nodes
    sd[0].primIds          = psc[0].prim_ids
    sd[0].meshes           = psc[0].meshes
    sd[0].meshCount        = Int64(psc[0].mesh_count)
    sd[0].materials        = psc[0].materials
    sd[0].materialCount    = Int64(psc[0].material_count)
    sd[0].areaLights       = psc[0].area_lights
    sd[0].areaLightCount   = Int64(psc[0].area_light_count)
    sd[0].textures         = psc[0].tex_filenames
    sd[0].textureCount     = Int64(psc[0].tex_count)
    sd[0].distantLights    = psc[0].distant_lights
    sd[0].distantLightCount = Int64(psc[0].distant_count)
    sd[0].pointLights      = psc[0].point_lights
    sd[0].pointLightCount  = Int64(psc[0].point_count)
    sd[0].infiniteLights   = psc[0].infinite_lights
    sd[0].infiniteLightCount = Int64(psc[0].infinite_count)
    sd[0].spheres          = psc[0].spheres
    sd[0].sphereCount      = Int64(psc[0].sphere_count)
    sd[0].mediums          = psc[0].mediums
    sd[0].mediumCount      = Int64(psc[0].medium_count)
    sd[0].mediumInterfaces = psc[0].medium_ifaces
    sd[0].mediumIfaceCount = Int64(psc[0].medium_iface_count)
    sd[0].lightSampler    = psc[0].light_sampler
    return sd
