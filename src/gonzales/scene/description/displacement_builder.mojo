"""pbrt's shape-level `"texture displacement"`: the mesh just stored becomes displaced triangles (PrimId.type 9)
over the texture's height map instead of pbrt's pre-tessellation to "edgelength" (see scene/shapes/displacement.mojo)."""
from std.math import sqrt, abs
from std.memory.alloc import unsafe_alloc
from gonzales.core.diagnostics import warn_unsupported_in
from gonzales.core.geometry import Vec3f, cross
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.description.lexer import ParameterDictionary
from gonzales.scene.description.parse_types import SceneParseState
from gonzales.scene.description.material_builder import _imagemap_affine
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
