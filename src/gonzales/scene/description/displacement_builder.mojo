"""pbrt's shape-level `"texture displacement"`: the mesh just stored becomes displaced triangles (PrimId.type 9)
over the texture's height map instead of pbrt's pre-tessellation to "edgelength" (see scene/shapes/displacement.mojo)."""
from std.math import sqrt, abs, tan, min, max
from std.os import getenv
from std.memory import bitcast
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc
from gonzales.core.diagnostics import warn_unsupported_in
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.description.lexer import ParameterDictionary
from gonzales.scene.description.parse_types import SceneParseState, MeshAccum
from gonzales.scene.description.material_builder import _imagemap_affine
from gonzales.scene.shapes.disp_build import DispOptions, DispMesh, load_disp_mesh, disp_bounds, empty_disp_mesh, in_cone
from gonzales.scene.shapes.displaced_patches import SubdivSurface
from gonzales.scene.shapes.displacement import DisplacedMesh, DisplacedTri, Rmip, displaced_tri, load_height_map


def _skip(name: String, why: String):
    warn_unsupported_in("displacement", why, "texture", name, "the shape renders undisplaced", "imagemap with uv mapping")


def apply_shape_displacement(s: Pointer[SceneParseState, MutUntrackedOrigin], params: ParameterDictionary):
    var name = params.get_string("displacement", "")
    if name == "":
        return
    ref ma = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
    if ma.is_area_light:
        _skip(name, "on an area light"); return
    var ti = len(s[unsafe_offset=0].tex_names) - 1
    while ti >= 0 and s[unsafe_offset=0].tex_names[ti] != name:
        ti -= 1
    if ti < 0:
        _skip(name, "not an imagemap"); return
    var m = GpuTexture.UV_IDENTITY
    for k in range(len(s[unsafe_offset=0].map_tex_idx)):
        if s[unsafe_offset=0].map_tex_idx[k] == ti:
            for c in range(16): m[c] = s[unsafe_offset=0].map_tex_map[16 * k + c]
    if Int(m[15]) != GpuTexture.MAP_UV:
        _skip(name, "a non-uv mapping"); return
    var nv = len(ma.points) // 4
    var nt = len(ma.vert_idxs) // 3
    if len(ma.uvs) < 2 * nv:
        _skip(name, "a mesh without uv"); return
    # pbrt displaces in object space; a uniform scale carries the heights along
    var c = s[unsafe_offset=0].ctm.copy()
    var det = c[0] * (c[5] * c[10] - c[9] * c[6]) - c[4] * (c[1] * c[10] - c[9] * c[2]) + c[8] * (c[1] * c[6] - c[5] * c[2])
    var k = abs(det) ** Float32(1.0 / 3.0)
    var aff = _imagemap_affine(s, ti)
    var file = s[unsafe_offset=0].tex_files[ti]
    var key = file + "|" + String(k * aff.scale.r) + "|" + String(k * aff.bias.r)
    var mi = -1
    for q in range(len(s[unsafe_offset=0].disp_map_keys)):
        if s[unsafe_offset=0].disp_map_keys[q] == key:
            mi = q
    if mi < 0:
        var loaded = load_height_map(file, k * aff.scale.r, k * aff.bias.r)
        if loaded.w == Int32(0):
            _skip(name, "an unreadable file"); return
        mi = len(s[unsafe_offset=0].disp_maps)
        s[unsafe_offset=0].disp_maps.append(loaded)
        s[unsafe_offset=0].disp_map_keys.append(key)
    var rm = s[unsafe_offset=0].disp_maps[mi]
    # displacement directions: the shape's normals before pbrt's ReverseOrientation flip, else area-weighted ones
    var sgn = Float32(-1) if s[unsafe_offset=0].cur_attr.reverse_orient else Float32(1)
    var nrm = List[Vec3f](length=nv, fill=Vec3f(0))
    var has_n = len(ma.normals) >= 3 * nv
    if not has_n:
        for t in range(nt):
            var i0 = Int(ma.vert_idxs[3 * t]); var i1 = Int(ma.vert_idxs[3 * t + 1]); var i2 = Int(ma.vert_idxs[3 * t + 2])
            var fa = cross(_pt(ma.points, i1) - _pt(ma.points, i0), _pt(ma.points, i2) - _pt(ma.points, i0))
            nrm[i0] += fa; nrm[i1] += fa; nrm[i2] += fa
    for v in range(nv):
        var n = Vec3f(ma.normals[3 * v], ma.normals[3 * v + 1], ma.normals[3 * v + 2]) if has_n else nrm[v]
        var l = n.length()
        nrm[v] = n * (sgn / l) if l > Float32(0) else Vec3f(0, 0, 1)
    var tw = Float32(rm.w); var th = Float32(rm.hh)
    @always_inline
    def texel(v: Int) {imm} -> SIMD[DType.float32, 2]:
        var su = m[0] * ma.uvs[2 * v] + m[2]
        var sv = Float32(1) - (m[1] * ma.uvs[2 * v + 1] + m[3])   # pbrt's image V flip
        return SIMD[DType.float32, 2](su * tw - Float32(0.5), sv * th - Float32(0.5))
    var tris = unsafe_alloc[DisplacedTri](max(nt, 1))
    for t in range(nt):
        var i0 = Int(ma.vert_idxs[3 * t]); var i1 = Int(ma.vert_idxs[3 * t + 1]); var i2 = Int(ma.vert_idxs[3 * t + 2])
        tris[unsafe_offset=t] = displaced_tri(_pt(ma.points, i0), _pt(ma.points, i1), _pt(ma.points, i2),
                                              nrm[i0], nrm[i1], nrm[i2], texel(i0), texel(i1), texel(i2), rm)
    ma.disp_idx = Int32(len(s[unsafe_offset=0].disp_meshes))
    s[unsafe_offset=0].disp_meshes.append(DisplacedMesh(tris, rm))


@always_inline
def _pt(p: List[Float32], v: Int) -> Vec3f:
    return Vec3f(p[4 * v], p[4 * v + 1], p[4 * v + 2])


def _world_eye(w2c: Array[Float32, 16]) -> Vec3f:
    """The camera position: the point the column-major world-to-camera matrix takes to the origin."""
    var a = List[Float64](length=12, fill=Float64(0))   # rows r: m[c*4 + r] for c < 3, then -m[12 + r]
    for r in range(3):
        for c in range(3):
            a[4 * r + c] = Float64(w2c[c * 4 + r])
        a[4 * r + 3] = -Float64(w2c[12 + r])
    for c in range(3):   # Gauss-Jordan with partial pivoting
        var p = c
        for r in range(c + 1, 3):
            if abs(a[4 * r + c]) > abs(a[4 * p + c]):
                p = r
        for k in range(4):
            var t = a[4 * c + k]; a[4 * c + k] = a[4 * p + k]; a[4 * p + k] = t
        var piv = a[4 * c + c]
        if piv == Float64(0):
            return Vec3f(0, 0, 0)
        for k in range(4):
            a[4 * c + k] /= piv
        for r in range(3):
            if r != c:
                var f = a[4 * r + c]
                for k in range(4):
                    a[4 * r + k] -= f * a[4 * c + k]
    return Vec3f(Float32(a[3]), Float32(a[7]), Float32(a[11]))


def subdiv_camera(s: Pointer[SceneParseState, MutUntrackedOrigin]) -> Tuple[Vec3f, Float32, Vec3f, Float32]:
    """The eye, a pixel's angle along the shorter film axis, the view direction and the cosine of half the diagonal view."""
    var w = Float32(s[unsafe_offset=0].film_w); var h = Float32(s[unsafe_offset=0].film_h)
    var short = min(w, h); var aspect = max(w, h) / max(short, Float32(1))
    var tan_h = tan(s[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0))
    ref m = s[unsafe_offset=0].cam2w_raw   # world to camera: the camera's +z is its rotation's third row
    var f = Vec3f(m[2], m[6], m[10]); f = f * (Float32(1) / sqrt(max(dot(f, f), Float32(1e-30))))
    var tan_d = tan_h * sqrt(Float32(1) + aspect * aspect)
    return (_world_eye(m), Float32(2) * tan_h / short, f, Float32(1) / sqrt(Float32(1) + tan_d * tan_d))


def _cage_layout(ma: MeshAccum) -> Bool:
    """Moana's Ptex cages: Ptex face f is triangles 2f, 2f + 1 with face uv (0,0) (1,0) (1,1) and (0,0) (1,1) (0,1)."""
    var nt = len(ma.vert_idxs) // 3
    if nt == 0 or nt % 2 != 0 or len(ma.ptex_faces) != nt or len(ma.uvs) < 2:
        return False
    var tu: List[Float32] = [0, 1, 1, 0, 1, 0]; var tv: List[Float32] = [0, 0, 1, 0, 1, 1]
    for t in range(nt):
        if Int(ma.ptex_faces[t]) != t // 2:
            return False
        for c in range(3):
            var vi = Int(ma.vert_idxs[3 * t + c])
            if abs(ma.uvs[2 * vi] - tu[3 * (t % 2) + c]) > Float32(1e-4) or abs(ma.uvs[2 * vi + 1] - tv[3 * (t % 2) + c]) > Float32(1e-4):
                return False
    return True


def _exag() -> Float32:
    """GONZALES_SUBDIV_SCALE: the heights times this (pictures only; default 1)."""
    try:
        if getenv("GONZALES_SUBDIV_SCALE").byte_length() > 0:
            return Float32(atof(getenv("GONZALES_SUBDIV_SCALE")))
    except:
        pass
    return Float32(1)


def subdiv_host_surface(s: Pointer[SceneParseState, MutUntrackedOrigin], k: Int) -> Pointer[SubdivSurface, MutUntrackedOrigin]:
    """Subdiv mesh k's surface for a TriangleMesh (host data; gpu/scene.mojo uploads it). Ptex bytes are heights / 255."""
    if k >= len(s[unsafe_offset=0].subdiv_meshes) or s[unsafe_offset=0].subdiv_meshes[k].np == 0:   # not loaded: the cage stays
        return Pointer[SubdivSurface, MutUntrackedOrigin].unsafe_dangling()
    var cam = subdiv_camera(s)
    var p = unsafe_alloc[SubdivSurface](1)
    p[unsafe_offset=0] = s[unsafe_offset=0].subdiv_meshes[k].subdiv(s[unsafe_offset=0].subdiv_scale[k] * _exag() / Float32(255), s[unsafe_offset=0].subdiv_eye[k], cam[1])
    return p


def apply_subdiv_displacement(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """GONZALES_SUBDIV_DISP=1: a mesh whose material has a Ptex "texture displacement" is that Ptex's cage; its displaced
    Catmull-Clark surface (scene/shapes/disp_build.mojo) loads after the parse, once per file (load_subdiv_meshes)."""
    if getenv("GONZALES_SUBDIV_DISP") != "1":
        return
    var mi = Int(s[unsafe_offset=0].cur_attr.mat_idx)
    if mi < 0 or mi >= len(s[unsafe_offset=0].named_materials):
        return
    var file = s[unsafe_offset=0].named_materials[mi].disp_ptex_file
    if file.byte_length() == 0:
        return
    ref ma = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
    if not _cage_layout(ma):
        print("subdiv displacement:", file, ": the mesh is not two triangles a Ptex face in face order - it keeps its bump map")
        return
    for i in range(16):   # the cage is in the Ptex's own coordinates
        if abs(s[unsafe_offset=0].ctm[i] - (Float32(1) if i % 5 == 0 else Float32(0))) > Float32(1e-6):
            print("subdiv displacement:", file, ": the shape is transformed - it keeps its bump map")
            return
    var nf = len(ma.vert_idxs) // 6
    var corners = List[Float32](capacity=12 * nf)   # face f: triangles (A, B, C), (A, C, D)
    var h = UInt64(14695981039346656037)
    for f in range(nf):
        var vs: List[Int] = [Int(ma.vert_idxs[6 * f]), Int(ma.vert_idxs[6 * f + 1]), Int(ma.vert_idxs[6 * f + 2]), Int(ma.vert_idxs[6 * f + 5])]
        for v in vs:
            for i in range(3):
                var x = ma.points[4 * v + i]   # x, y, z, 1 a vertex (store_mesh)
                corners.append(x)
                h = (h ^ UInt64(bitcast[DType.uint32](x))) * UInt64(1099511628211)
    var k = -1   # one surface per file and pose
    for i in range(len(s[unsafe_offset=0].subdiv_files)):
        if s[unsafe_offset=0].subdiv_hash[i] == h and s[unsafe_offset=0].subdiv_files[i] == file and s[unsafe_offset=0].subdiv_corners[i] == corners:
            k = i
            break
    if k < 0:
        s[unsafe_offset=0].subdiv_files.append(file)
        s[unsafe_offset=0].subdiv_scale.append(s[unsafe_offset=0].named_materials[mi].disp_ptex_scale)
        s[unsafe_offset=0].subdiv_corners.append(corners^)
        s[unsafe_offset=0].subdiv_hash.append(h)
        k = len(s[unsafe_offset=0].subdiv_files) - 1
    ma.subdiv_idx = Int32(k)


def _xform(m: List[Float32], o: Int, p: Vec3f) -> Vec3f:
    """Column-major 4 x 4 matrix m[o .. o + 16] times point p."""
    return Vec3f(m[o] * p.x + m[o + 4] * p.y + m[o + 8] * p.z + m[o + 12], m[o + 1] * p.x + m[o + 5] * p.y + m[o + 9] * p.z + m[o + 13],
                 m[o + 2] * p.x + m[o + 6] * p.y + m[o + 10] * p.z + m[o + 14])


def _box_dist(p: Vec3f, b: Tuple[Vec3f, Vec3f]) -> Float32:
    var dx = max(max(b[0].x - p.x, p.x - b[1].x), Float32(0)); var dy = max(max(b[0].y - p.y, p.y - b[1].y), Float32(0))
    var dz = max(max(b[0].z - p.z, p.z - b[1].z), Float32(0))
    return sqrt(dx * dx + dy * dy + dz * dz)


def _view_dist(e: Vec3f, f: Vec3f, cos_cone: Float32, b: Tuple[Vec3f, Vec3f], far: Float32) -> Float32:
    """The box's distance from eye e, times far when it is outside the view cone."""
    var c = (b[0] + b[1]) * Float32(0.5); var x = b[1] - b[0]
    var d = _box_dist(e, b)
    return d if in_cone(e, f, cos_cone, c, Float32(0.5) * sqrt(dot(x, x))) else d * far


def load_subdiv_meshes(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """Loads every recorded subdiv file for the eye nearest to its cage among the instances that place it (in their object
    space; the world for a mesh outside an object): that instance needs the finest heights and sub-squares."""
    var n = len(s[unsafe_offset=0].subdiv_files)
    if n == 0 or len(s[unsafe_offset=0].subdiv_meshes) == n:
        return
    var cam = subdiv_camera(s)
    var boxes = List[Tuple[Vec3f, Vec3f]]()   # of each one's corners
    for k in range(n):
        ref c = s[unsafe_offset=0].subdiv_corners[k]
        var lo = Vec3f(3e38, 3e38, 3e38); var hi = Vec3f(-3e38, -3e38, -3e38)
        for j in range(len(c) // 3):
            lo = Vec3f(min(lo.x, c[3 * j]), min(lo.y, c[3 * j + 1]), min(lo.z, c[3 * j + 2]))
            hi = Vec3f(max(hi.x, c[3 * j]), max(hi.y, c[3 * j + 1]), max(hi.z, c[3 * j + 2]))
        boxes.append((lo, hi))
    var best_d = List[Float32](length=n, fill=Float32(3e38)); var best_e = List[Vec3f](length=n, fill=cam[0])
    var best_f = List[Vec3f](length=n, fill=cam[2])
    comptime FAR = Float32(16)   # outside the view a cage counts this much farther: secondary rays only
    var nt = len(s[unsafe_offset=0].object_mesh_start)
    var files_of = List[List[Int]]()   # per template the files of its meshes
    for _ in range(nt):
        files_of.append(List[Int]())
    for i in range(len(s[unsafe_offset=0].meshes)):
        var k = Int(s[unsafe_offset=0].meshes[i].subdiv_idx)
        if k < 0:
            continue
        var t = -1
        for j in range(nt):
            if Int(s[unsafe_offset=0].object_mesh_start[j]) <= i and i < Int(s[unsafe_offset=0].object_mesh_end[j]):
                t = j
                break
        if t < 0:   # in the world
            var d = _view_dist(cam[0], cam[2], cam[3], boxes[k], FAR)
            if d < best_d[k]:
                best_d[k] = d; best_e[k] = cam[0]; best_f[k] = cam[2]
        elif k not in files_of[t]:
            files_of[t].append(k)
    for j in range(len(s[unsafe_offset=0].instance_template_idx)):
        var t = Int(s[unsafe_offset=0].instance_template_idx[j])
        if t < 0 or t >= nt or len(files_of[t]) == 0:
            continue
        ref wo = s[unsafe_offset=0].instance_world_to_obj
        var e = _xform(wo, 16 * j, cam[0])
        var o = 16 * j
        var f = Vec3f(wo[o] * cam[2].x + wo[o + 4] * cam[2].y + wo[o + 8] * cam[2].z, wo[o + 1] * cam[2].x + wo[o + 5] * cam[2].y + wo[o + 9] * cam[2].z,
                      wo[o + 2] * cam[2].x + wo[o + 6] * cam[2].y + wo[o + 10] * cam[2].z)
        f = f * (Float32(1) / sqrt(max(dot(f, f), Float32(1e-30))))
        for k in files_of[t]:
            var d = _view_dist(e, f, cam[3], boxes[k], FAR)
            if d < best_d[k]:
                best_d[k] = d; best_e[k] = e; best_f[k] = f
    var lod = Float32(0.25); var pps = Float32(0.5); var minpx = Float32(1); var eps = Float32(0); var tps = Float32(5)
    try:
        if getenv("GONZALES_SUBDIV_MINPX").byte_length() > 0:
            minpx = Float32(atof(getenv("GONZALES_SUBDIV_MINPX")))
        if getenv("GONZALES_SUBDIV_LOD").byte_length() > 0:
            lod = Float32(atof(getenv("GONZALES_SUBDIV_LOD")))
        if getenv("GONZALES_SUBDIV_PPS").byte_length() > 0:
            pps = Float32(atof(getenv("GONZALES_SUBDIV_PPS")))
        if getenv("GONZALES_SUBDIV_EXACT_PX").byte_length() > 0:
            eps = Float32(atof(getenv("GONZALES_SUBDIV_EXACT_PX")))
        if getenv("GONZALES_SUBDIV_TPS").byte_length() > 0:
            tps = Float32(atof(getenv("GONZALES_SUBDIV_TPS")))
    except:
        pass
    var uses = List[Int](length=n, fill=0)
    for i in range(len(s[unsafe_offset=0].meshes)):
        if s[unsafe_offset=0].meshes[i].subdiv_idx >= Int32(0):
            uses[Int(s[unsafe_offset=0].meshes[i].subdiv_idx)] += 1
    var skipped = 0
    for k in range(n):
        var file = s[unsafe_offset=0].subdiv_files[k]
        s[unsafe_offset=0].subdiv_eye.append(best_e[k])
        # Heights reach the scale at most (Ptex bytes / 255): under minpx pixels where it is nearest (out of view: FAR times
        # farther) the displacement does not show, and the mesh keeps its bump map.
        if s[unsafe_offset=0].subdiv_scale[k] * _exag() < minpx * cam[1] * best_d[k]:
            s[unsafe_offset=0].subdiv_meshes.append(empty_disp_mesh()); skipped += 1
            continue
        var t0 = perf_counter_ns()
        try:
            var m = load_disp_mesh(file, DispOptions(best_e[k], best_f[k], cam[3], FAR, cam[1], lod, pps, tps, 2, -1, True, eps,
                                               s[unsafe_offset=0].subdiv_scale[k] * _exag() / Float32(255)),
                                   Int(s[unsafe_offset=0].subdiv_corners[k].unsafe_ptr()))
            var mb = Float64(m.nchain + m.npyr + 4 * m.ncg + m.np * (64 * 3 + 12 + 4 + 48 + 1)) / Float64(1 << 20)
            print("subdiv displacement:", file, ":", uses[k], "meshes,", m.np, "patches,", m.prims(), "sub-squares,", m.texels, "texels kept,",
                  Int(mb), "MiB host data,", Float64(Int(perf_counter_ns()) - Int(t0)) * 1e-9, "s")
            s[unsafe_offset=0].subdiv_meshes.append(m^)
        except err:
            print("subdiv displacement:", file, ":", err)
            s[unsafe_offset=0].subdiv_meshes.append(empty_disp_mesh())
    print("subdiv displacement:", n - skipped, "of", n, "surfaces displaced; the others keep their bump maps (under",
          minpx, "pixel of relief)")
