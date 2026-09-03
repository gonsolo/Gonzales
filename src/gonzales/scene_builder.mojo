from std.memory import alloc
from .parse_types import SceneParseState, MeshAccum
from .transform import transform_points

# ── Shared scene-building API ───────────────────────────────────────────────
#
# Functions here append fully-resolved geometry/materials/lights into a
# `SceneParseState` and have no pbrt-directive-specific logic in them -- any
# scene-format front-end (pbrt's own directive handlers in pbrt_parser.mojo,
# or a future mitsuba_parser.mojo) can call them directly. `SceneParseState`/
# `MeshAccum`/`NamedMaterial` are plain public structs with plain `List[T]`
# fields, so most other scene data (materials, non-area lights, media) needs
# no wrapper at all -- a front-end can append to those lists itself. This
# file exists for the pieces with real transform/construction logic worth
# not reimplementing per format.

def store_mesh[Of: Origin[mut=True], Oi: Origin[mut=True]](
    s:       UnsafePointer[SceneParseState, MutExternalOrigin],
    tmp_f:   UnsafePointer[Float32, Of],
    tmp_i:   UnsafePointer[Int32, Oi],
    n_verts: Int32,
    n_tris:  Int32,
):
    """Append one triangle mesh: object-space points (3 floats/vertex) +
    flat triangle indices, transformed by the current CTM (`s[0].ctm`) and
    tagged with the current material/area-light/medium state
    (`s[0].cur_attr`) -- the same "resolve to final CTM/material, then
    append" contract any scene-format front-end must satisfy before calling
    this. UV/normal data (optional) is appended separately by the caller,
    directly onto `s[0].meshes[len(s[0].meshes) - 1]` after this returns."""
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
