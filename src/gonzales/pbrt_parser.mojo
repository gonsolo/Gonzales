from std.ffi import external_call
from std.time import perf_counter_ns
from std.memory import alloc
from std.math import tan, sqrt, abs
from std.subprocess import run
from std.os.path import exists
from .lexer import (PbrtScanner, scanner_open, scanner_free, scanner_is_at_end,
                    scanner_scan_token, scanner_parse_quoted_string,
                    scanner_scan_char, scanner_scan_float,
                    _psc_streq,
                    _psc_scan_spectrum_scalar, _psc_collect_params, ParameterDictionary,
                    _psc_skip_params, _psc_skip_line)
from .parse_types import (SceneParseState, MeshAccum, NamedMaterial,
                           ctm_push, ctm_pop, PSC_NAME_MAX, PSC_FILE_MAX)
from .geometry import (RGB, SampledSpectrum, Point3f, Vec3f, Material_C, MatKind, AreaLight_C,
                        Sphere_C, Curve_C, CURVE_N_PIECES, curve_piece_bounds, curve_bspline_point, curve_light_tube_area, dot, DistantLight_C, PointLight_C, InfiniteLight_C,
                        TriangleMesh_C, PrimId_C, Medium_C, MediumInterface_C, Grid_C, PI,
                        LightSampler_C, Instance_C, MeasuredBRDF_C, GpuTexture_C, _is_real_ptr)
from .transform import matrix_multiply, matrix_invert, transform_points, transform_normals
from .bvh import BVH2Node, SceneDescriptor2_C, build_bvh2
from .spectrum import SpectralHandle
from .sampling import gaussian_norm
from .ply import load_ply
from .material_builder import _psc_handle_make_named_material, _psc_handle_named_material
from .measured_bsdf import load_measured_brdf_full
from .light_builder import _psc_handle_area_light_source, handle_light_source
from .scene_builder import store_mesh

# ── Output struct ─────────────────────────────────────────────────────────────

struct ParsedScene_Mojo:
    var raster_to_camera: UnsafePointer[Float32, MutExternalOrigin]   # 16 floats, column-major
    var camera_to_world:  UnsafePointer[Float32, MutExternalOrigin]   # 16 floats, column-major
    var materials:        UnsafePointer[Material_C, MutExternalOrigin]
    var material_count:   Int32
    var area_lights:      UnsafePointer[AreaLight_C, MutExternalOrigin]
    var area_light_count: Int32
    var meshes:           UnsafePointer[TriangleMesh_C, MutExternalOrigin]
    var mesh_pts:         UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin]
    var mesh_vis:         UnsafePointer[UnsafePointer[Int64, MutExternalOrigin], MutExternalOrigin]
    var mesh_fis:         UnsafePointer[UnsafePointer[Int64, MutExternalOrigin], MutExternalOrigin]
    var mesh_n_verts:     UnsafePointer[Int32, MutExternalOrigin]
    var mesh_n_tris:      UnsafePointer[Int32, MutExternalOrigin]
    var mesh_uv_n_verts:  UnsafePointer[Int32, MutExternalOrigin]  # per-mesh UV vertex count; 0 = no UVs
    var mesh_nrm_n_verts: UnsafePointer[Int32, MutExternalOrigin]  # per-mesh normal vertex count; 0 = no shading normals
    var mesh_count:       Int32
    var bvh_nodes:        UnsafePointer[BVH2Node, MutExternalOrigin]   # GPU-safe TLAS: tris+curves only, no instance leaves
    var prim_ids:         UnsafePointer[PrimId_C, MutExternalOrigin]
    var bvh_node_count:   Int32
    var prim_count:       Int32
    # CPU-inclusive TLAS: tris+curves+instances. Used only by
    # mojo_parsed_scene_descriptor (SceneDescriptor2_C, the CPU render path).
    # GPU's device-side upload always reads bvh_nodes/prim_ids above instead —
    # its traversal kernels have no BLAS/instance buffers to resolve a
    # PrimId_C.type==6 leaf, so one must never appear in its uploaded arrays
    # (confirmed via testing: it does not degrade gracefully, it crashes).
    var bvh_nodes_cpu:      UnsafePointer[BVH2Node, MutExternalOrigin]
    var prim_ids_cpu:       UnsafePointer[PrimId_C, MutExternalOrigin]
    var bvh_node_count_cpu: Int32
    var prim_count_cpu:     Int32
    var film_w:           Int32
    var film_h:           Int32
    # Film "float cropwindow" [x0 x1 y0 y1] — normalized fractional bounds
    # of film_w/film_h to actually render/output. Defaults to (0,1,0,1),
    # the full frame, when the scene doesn't specify one.
    var crop_x0: Float32
    var crop_x1: Float32
    var crop_y0: Float32
    var crop_y1: Float32
    var camera_fov:       Float32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    UnsafePointer[UInt8, MutExternalOrigin]      # null-terminated
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
    var sppm_radius:            Float32  # -1 = not specified by scene; caller falls back to CLI/default
    var sppm_photons_per_iter:  Int32    # -1 = not specified by scene; pbrt itself defaults to film_w*film_h
    var tex_filenames:    UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin]
    var tex_count:        Int32
    var distant_lights:   UnsafePointer[DistantLight_C, MutExternalOrigin]
    var distant_count:    Int32
    var point_lights:     UnsafePointer[PointLight_C, MutExternalOrigin]
    var point_count:      Int32
    var infinite_lights:  UnsafePointer[InfiniteLight_C, MutExternalOrigin]
    var infinite_count:   Int32
    var spheres:          UnsafePointer[Sphere_C, MutExternalOrigin]
    var sphere_count:     Int32
    var curves:           UnsafePointer[Curve_C, MutExternalOrigin]
    var curve_count:      Int32
    var mediums:          UnsafePointer[Medium_C, MutExternalOrigin]
    var medium_count:     Int32
    var medium_ifaces:    UnsafePointer[MediumInterface_C, MutExternalOrigin]
    var medium_iface_count: Int32
    var grids:            UnsafePointer[Grid_C, MutExternalOrigin]
    var grid_count:       Int32
    var light_sampler:    LightSampler_C
    # Object instancing: one BLAS (private BVH2, over `meshes` above) per
    # ObjectBegin/ObjectEnd template, referenced by Instance_C.blasIdx.
    var blas_nodes_arr:   UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin]
    var blas_primids_arr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin]
    var blas_node_counts:   UnsafePointer[Int32, MutExternalOrigin]  # per-BLAS array length, needed for GPU upload
    var blas_primid_counts: UnsafePointer[Int32, MutExternalOrigin]
    var blas_count:       Int32
    var instances:        UnsafePointer[Instance_C, MutExternalOrigin]
    var instance_count:   Int32
    # Mesh-index range [start, end) each template's BLAS spans, into the SAME
    # `meshes` array above (a template can bundle several Shape calls, e.g.
    # barcelona-pavilion's tree templates each have 5-9 plymesh shapes).
    # Only consumer today: the Vulkan RT interop scene builder (pipeline.mojo),
    # which needs to know which mesh indices are template-only (excluded from
    # its ordinary one-BLAS-per-mesh loop) and which meshes feed which
    # per-template multi-geometry BLAS -- see [[project_vulkan_rt_backend]].
    var template_mesh_start: UnsafePointer[Int32, MutExternalOrigin]
    var template_mesh_end:   UnsafePointer[Int32, MutExternalOrigin]
    # "measured" materials: one MeasuredBRDF_C per distinct .bsdf file
    # (deduped by path), referenced by Material_C.measured_idx. Populated at
    # final-scene-build time from named_materials[i].measured_bsdf_path -- see
    # the dedup+load loop near the materials array build below.
    var measured_brdfs:  UnsafePointer[MeasuredBRDF_C, MutExternalOrigin]
    var measured_count:  Int32

# ── Matrix utilities ──────────────────────────────────────────────────────────

def _psc_identity(m: UnsafePointer[Float32, MutExternalOrigin]):
    for i in range(16):
        m[i] = Float32(0)
    m[0] = Float32(1)
    m[5] = Float32(1)
    m[10] = Float32(1)
    m[15] = Float32(1)

def _psc_matcopy(dst: UnsafePointer[Float32, MutExternalOrigin],
                src: UnsafePointer[Float32, MutExternalOrigin]):
    for i in range(16):
        dst[i] = src[i]

def _psc_ctm_concat(s: UnsafePointer[SceneParseState, MutExternalOrigin],
                   t: UnsafePointer[Float32, MutExternalOrigin]):
    """Compute s.ctm = s.ctm × t and store back."""
    var result = alloc[Float32](16)
    matrix_multiply(s[0].ctm.unsafe_ptr(), t, result)
    for i in range(16):
        s[0].ctm[i] = result[i]
    result.free()

def _psc_row_to_col(col_out: UnsafePointer[Float32, MutExternalOrigin],
                   row_in:  UnsafePointer[Float32, MutExternalOrigin]):
    for row in range(4):
        for col in range(4):
            col_out[col * 4 + row] = row_in[row * 4 + col]

# ── Transform keyword handlers ────────────────────────────────────────────────

def _psc_handle_translate(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                         s: UnsafePointer[SceneParseState, MutExternalOrigin]):
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

def _psc_handle_scale_kw(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                        s: UnsafePointer[SceneParseState, MutExternalOrigin]):
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

def _psc_handle_rotate(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                      s: UnsafePointer[SceneParseState, MutExternalOrigin]):
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

def _psc_handle_lookat(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                      s: UnsafePointer[SceneParseState, MutExternalOrigin]):
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

def _psc_handle_integrator(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                          s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.free()
    var params = _psc_collect_params(handle)
    s[0].max_depth = params.get_int("maxdepth", s[0].max_depth)
    s[0].sppm_radius = params.get_float("radius", s[0].sppm_radius)
    s[0].sppm_photons_per_iter = params.get_int("photonsperiteration", s[0].sppm_photons_per_iter)

def _psc_handle_sampler(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                       s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.free()
    var params = _psc_collect_params(handle)
    s[0].samples_per_pixel = params.get_int("pixelsamples", s[0].samples_per_pixel)
    s[0].samples_per_pixel = params.get_int("samples", s[0].samples_per_pixel)

def _psc_handle_filter(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                      s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    if _psc_streq(sbuf, "triangle") or _psc_streq(sbuf, "tent"):
        s[0].filter_type = Int32(1)
    elif _psc_streq(sbuf, "box"):
        s[0].filter_type = Int32(2)
    else:
        s[0].filter_type = Int32(0)  # gaussian (default)
    sbuf.free()
    var params = _psc_collect_params(handle)
    s[0].filter_support_x = params.get_float("xradius", s[0].filter_support_x)
    s[0].filter_support_y = params.get_float("yradius", s[0].filter_support_y)
    s[0].filter_sigma = params.get_float("sigma", s[0].filter_sigma)

def _psc_handle_film(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                    s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.free()
    var params = _psc_collect_params(handle)
    s[0].film_w = params.get_int("xresolution", s[0].film_w)
    s[0].film_h = params.get_int("yresolution", s[0].film_h)
    s[0].film_filename = params.get_string("filename", s[0].film_filename)
    s[0].film_iso = params.get_float("iso", s[0].film_iso)
    s[0].film_max_comp = params.get_float("maxcomponentvalue", s[0].film_max_comp)
    var cw = params.get_floats("cropwindow")
    if len(cw) >= 4:
        s[0].crop_x0 = cw[0]; s[0].crop_x1 = cw[1]
        s[0].crop_y0 = cw[2]; s[0].crop_y1 = cw[3]

def _psc_handle_camera(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                      s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.free()
    # Copy current CTM into cam2w_raw
    for i in range(16): s[0].cam2w_raw[i] = s[0].ctm[i]
    var params = _psc_collect_params(handle)
    s[0].camera_fov = params.get_float("fov", s[0].camera_fov)

def _psc_handle_transform(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                         s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    _ = scanner_scan_char(handle, UInt8(91))  # '['
    var tmp = alloc[Float32](1)
    for i in range(16):
        _ = scanner_scan_float(handle, tmp)
        s[0].ctm[i] = tmp[0]
    tmp.free()
    _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_handle_world_begin(s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    for i in range(16): s[0].ctm[i] = Float32(0)
    s[0].ctm[0] = Float32(1); s[0].ctm[5] = Float32(1)
    s[0].ctm[10] = Float32(1); s[0].ctm[15] = Float32(1)
    s[0].ctm_stack.clear()

def _psc_handle_attribute_begin(s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    ctm_push(s[0])
    s[0].attr_stack.append(s[0].cur_attr)

def _psc_handle_attribute_end(s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    ctm_pop(s[0])
    if len(s[0].attr_stack) > 0:
        s[0].cur_attr = s[0].attr_stack[len(s[0].attr_stack) - 1]
        _ = s[0].attr_stack.pop()

# ── Loop subdivision surface tessellation (task #159) ─────────────────────────
# `Shape "loopsubdiv"` (unlike trianglemesh/plymesh) gives a COARSE control
# mesh ("point3 P" + "integer indices") plus an "integer levels" subdivision
# count -- pbrt refines it via Loop's (1987) triangle subdivision scheme
# before rendering. Previously this shape type fell through handle_shape's
# catch-all `_psc_skip_params` (any type that isn't trianglemesh/plymesh/
# curve/sphere is silently dropped, no warning) -- found while investigating
# contemporary-bathroom's missing bathtub shell (its "bathtube" material's
# shape is a loopsubdiv control mesh with no plymesh fallback).

def _loopsubdiv_edge_index(
    mut edge_key_to_idx: Dict[Int64, Int32],
    mut edge_v0: List[Int32], mut edge_v1: List[Int32],
    mut edge_opp0: List[Int32], mut edge_opp1: List[Int32],
    mut edge_tri_count: List[Int32],
    va: Int32, vb: Int32, vopp: Int32,
) -> Int32:
    """Look up (or create) the undirected-edge record for (va,vb), and
    record `vopp` (the triangle's third vertex) as that edge's 1st or 2nd
    incident-triangle opposite vertex -- the two "opposite" vertices are
    what the interior odd-vertex mask (3/8, 3/8, 1/8, 1/8) needs. A 3rd+
    incident triangle (non-manifold edge) just overwrites edge_opp1,
    keeping only the first two -- a defensive fallback, not expected for a
    well-formed subdivision control mesh."""
    var lo = va if va < vb else vb
    var hi = vb if va < vb else va
    var key = Int64(lo) * Int64(1_000_000) + Int64(hi)
    var existing = edge_key_to_idx.get(key, Int32(-1))
    if existing >= Int32(0):
        edge_opp1[Int(existing)] = vopp
        edge_tri_count[Int(existing)] = Int32(2)
        return existing
    var idx = Int32(len(edge_v0))
    edge_key_to_idx[key] = idx
    edge_v0.append(lo); edge_v1.append(hi)
    edge_opp0.append(vopp); edge_opp1.append(Int32(-1))
    edge_tri_count.append(Int32(1))
    return idx

def _loopsubdiv_one_level(
    p_in: List[Float32], i_in: List[Int32],
) -> Tuple[List[Float32], List[Int32]]:
    """One level of Loop (1987) triangle subdivision: p_in is n_verts*3 flat
    object-space positions, i_in is n_tris*3 flat vertex indices. Returns a
    refined mesh with (n_verts + n_edges) vertices and n_tris*4 triangles.
    Interior edge midpoints (odd vertices) use the classic 3/8-3/8-1/8-1/8
    mask against the edge's two endpoints + the two triangles' opposite
    vertices; boundary edges (only one incident triangle) use a plain
    midpoint. Existing (even) vertices are repositioned: interior vertices
    via beta=3/(8n) (3/16 for valence n=3 -- pbrt's own LoopSubdiv::beta
    special-cases this the same way) against all neighbors; boundary
    vertices via the 3/4-1/8-1/8 mask against ONLY their two boundary-edge
    neighbors."""
    var n_verts = len(p_in) // 3
    var n_tris = len(i_in) // 3

    var edge_key_to_idx = Dict[Int64, Int32]()
    var edge_v0 = List[Int32]()
    var edge_v1 = List[Int32]()
    var edge_opp0 = List[Int32]()
    var edge_opp1 = List[Int32]()
    var edge_tri_count = List[Int32]()
    var tri_edge0 = List[Int32]()
    var tri_edge1 = List[Int32]()
    var tri_edge2 = List[Int32]()

    for t in range(n_tris):
        var v0 = i_in[t*3+0]
        var v1 = i_in[t*3+1]
        var v2 = i_in[t*3+2]
        var e0 = _loopsubdiv_edge_index(edge_key_to_idx, edge_v0, edge_v1, edge_opp0, edge_opp1, edge_tri_count, v0, v1, v2)
        var e1 = _loopsubdiv_edge_index(edge_key_to_idx, edge_v0, edge_v1, edge_opp0, edge_opp1, edge_tri_count, v1, v2, v0)
        var e2 = _loopsubdiv_edge_index(edge_key_to_idx, edge_v0, edge_v1, edge_opp0, edge_opp1, edge_tri_count, v2, v0, v1)
        tri_edge0.append(e0); tri_edge1.append(e1); tri_edge2.append(e2)

    var n_edges = len(edge_v0)

    # Vertex adjacency (all neighbors, for interior smoothing) + the two
    # boundary-edge neighbors specifically (for boundary smoothing).
    var vert_neighbors = List[List[Int32]]()
    var vert_boundary_a = List[Int32]()
    var vert_boundary_b = List[Int32]()
    for _ in range(n_verts):
        vert_neighbors.append(List[Int32]())
        vert_boundary_a.append(Int32(-1))
        vert_boundary_b.append(Int32(-1))
    for e in range(n_edges):
        var a = edge_v0[e]
        var b = edge_v1[e]
        vert_neighbors[Int(a)].append(b)
        vert_neighbors[Int(b)].append(a)
        if edge_tri_count[e] == Int32(1):
            if vert_boundary_a[Int(a)] < Int32(0):
                vert_boundary_a[Int(a)] = b
            else:
                vert_boundary_b[Int(a)] = b
            if vert_boundary_a[Int(b)] < Int32(0):
                vert_boundary_a[Int(b)] = a
            else:
                vert_boundary_b[Int(b)] = a

    var out_p = List[Float32]()
    out_p.reserve((n_verts + n_edges) * 3)

    # Even (repositioned original) vertices, same index order as input.
    for v in range(n_verts):
        var px = p_in[v*3+0]; var py = p_in[v*3+1]; var pz = p_in[v*3+2]
        if vert_boundary_a[v] >= Int32(0) and vert_boundary_b[v] >= Int32(0):
            var ba = Int(vert_boundary_a[v]); var bb = Int(vert_boundary_b[v])
            out_p.append(Float32(0.75)*px + Float32(0.125)*(p_in[ba*3+0] + p_in[bb*3+0]))
            out_p.append(Float32(0.75)*py + Float32(0.125)*(p_in[ba*3+1] + p_in[bb*3+1]))
            out_p.append(Float32(0.75)*pz + Float32(0.125)*(p_in[ba*3+2] + p_in[bb*3+2]))
        else:
            var n = len(vert_neighbors[v])
            if n == 0:
                out_p.append(px); out_p.append(py); out_p.append(pz)
            else:
                var beta: Float32
                if n == 3:
                    beta = Float32(3.0) / Float32(16.0)
                else:
                    beta = Float32(3.0) / (Float32(8.0) * Float32(n))
                var sx = Float32(0); var sy = Float32(0); var sz = Float32(0)
                for k in range(n):
                    var nb = Int(vert_neighbors[v][k])
                    sx += p_in[nb*3+0]; sy += p_in[nb*3+1]; sz += p_in[nb*3+2]
                var w = Float32(1) - Float32(n)*beta
                out_p.append(w*px + beta*sx)
                out_p.append(w*py + beta*sy)
                out_p.append(w*pz + beta*sz)

    # Odd (new edge-midpoint) vertices, index n_verts + e.
    for e in range(n_edges):
        var a = Int(edge_v0[e]); var b = Int(edge_v1[e])
        if edge_tri_count[e] == Int32(2):
            var o0 = Int(edge_opp0[e]); var o1 = Int(edge_opp1[e])
            out_p.append(Float32(0.375)*(p_in[a*3+0]+p_in[b*3+0]) + Float32(0.125)*(p_in[o0*3+0]+p_in[o1*3+0]))
            out_p.append(Float32(0.375)*(p_in[a*3+1]+p_in[b*3+1]) + Float32(0.125)*(p_in[o0*3+1]+p_in[o1*3+1]))
            out_p.append(Float32(0.375)*(p_in[a*3+2]+p_in[b*3+2]) + Float32(0.125)*(p_in[o0*3+2]+p_in[o1*3+2]))
        else:
            out_p.append(Float32(0.5)*(p_in[a*3+0]+p_in[b*3+0]))
            out_p.append(Float32(0.5)*(p_in[a*3+1]+p_in[b*3+1]))
            out_p.append(Float32(0.5)*(p_in[a*3+2]+p_in[b*3+2]))

    # New connectivity: each original triangle (v0,v1,v2) -> 4 new triangles,
    # via its 3 edge midpoints (m01,m12,m20), preserving winding order.
    var out_i = List[Int32]()
    out_i.reserve(n_tris * 4 * 3)
    for t in range(n_tris):
        var v0 = i_in[t*3+0]; var v1 = i_in[t*3+1]; var v2 = i_in[t*3+2]
        var m01 = Int32(n_verts) + tri_edge0[t]
        var m12 = Int32(n_verts) + tri_edge1[t]
        var m20 = Int32(n_verts) + tri_edge2[t]
        out_i.append(v0); out_i.append(m01); out_i.append(m20)
        out_i.append(v1); out_i.append(m12); out_i.append(m01)
        out_i.append(v2); out_i.append(m20); out_i.append(m12)
        out_i.append(m01); out_i.append(m12); out_i.append(m20)

    return (out_p^, out_i^)

def _loopsubdiv_tessellate(
    var p_in: List[Float32], var i_in: List[Int32], levels: Int32,
) -> Tuple[List[Float32], List[Int32]]:
    """Run `levels` iterations of _loopsubdiv_one_level, refining a Loop
    subdivision control mesh into the final triangle mesh passed to
    store_mesh -- same object-space contract trianglemesh's own "P"/
    "indices" params already have (CTM world-transform happens in
    store_mesh, not here)."""
    for _ in range(Int(levels)):
        var next_level = _loopsubdiv_one_level(p_in, i_in)
        p_in = next_level[0].copy()
        i_in = next_level[1].copy()
    return (p_in^, i_in^)

# ── Hair curve helpers ────────────────────────────────────────────────────────

def handle_curve_shape(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                            s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    """PBRT `Shape "curve"`: stored natively (no tessellation) as one Curve_C
    per local cubic B-spline segment, CTM-transformed at parse time. See
    Curve_C / intersect_curve in geometry.mojo for the BVH-time intersection."""
    var params = _psc_collect_params(handle)
    # take_floats moves "P"'s buffer out of the dictionary (List.pop, O(1) --
    # no copy); _psc_collect_params already scanned it straight into the
    # List's own backing buffer sized to exactly what's needed, matching the
    # old hand-rolled cp_buf's cost with none of its 512-point-default waste
    # (hair scenes have thousands of curve directives, most with a handful
    # of control points).
    var cp_list = params.take_floats("P")
    var n_cp = Int32(len(cp_list) / 3)
    var width0 = params.get_float("width0", params.get_float("width", Float32(0.002)))
    var width1 = params.get_float("width1", width0)
    if n_cp < Int32(4):
        return

    var n_raw = Int(n_cp)
    var raw4 = alloc[Float32](n_raw * 4)
    var xfm4 = alloc[Float32](n_raw * 4)
    for i in range(n_raw):
        raw4[i*4+0] = cp_list[i*3+0]; raw4[i*4+1] = cp_list[i*3+1]
        raw4[i*4+2] = cp_list[i*3+2]; raw4[i*4+3] = Float32(1)
    transform_points(s[0].ctm.unsafe_ptr(), raw4, Int32(n_raw), xfm4)
    raw4.free()

    # Split into (n_cp - 3) local B-spline segments: window i uses raw
    # control points [i, i+1, i+2, i+3] — matches the standard uniform
    # cubic B-spline curve-chain convention (same windowing PBRT itself uses).
    var n_seg = n_raw - 3
    var mat_idx = s[0].cur_attr.mat_idx
    for seg in range(n_seg):
        var t0 = Float32(seg) / Float32(n_seg)
        var t1 = Float32(seg + 1) / Float32(n_seg)
        for k in range(4):
            var vi = seg + k
            s[0].curves_cp.append(xfm4[vi*4+0])
            s[0].curves_cp.append(xfm4[vi*4+1])
            s[0].curves_cp.append(xfm4[vi*4+2])
        s[0].curves_w0.append(width0 + (width1 - width0) * t0)
        s[0].curves_w1.append(width0 + (width1 - width0) * t1)
        s[0].curves_mat.append(mat_idx)
        s[0].curves_al.append(s[0].cur_attr.is_alight)
        s[0].curves_al_rgb.append(s[0].cur_attr.al_rgb)
    xfm4.free()

# ── Medium handlers ───────────────────────────────────────────────────────────

def handle_named_medium(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                                  s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var name_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, name_buf, 64)
    var params = _psc_collect_params(handle)

    # "string type" used to need its own hand-written bracket-close
    # (["uniformgrid"] form) -- every other branch already consumed its
    # closing ']' when an array, this one didn't, silently truncating the
    # rest of the block (density/nx/ny/nz/sigma_a/sigma_s all skipped) for
    # any medium that happened to bracket-wrap "type". _psc_collect_params's
    # string-type handling always closes the bracket, so that bug class is
    # gone structurally, not just for this one param.
    var type_str = params.get_string("type", "")
    var is_hom = type_str == "homogeneous"
    var is_grid = type_str == "uniformgrid"

    # sigma_a/sigma_s: rgb triple, OR inline numeric spectrum array (mean of
    # samples, replicated to all 3 channels) via the same float-or-rgb
    # duality as texture "value"/"tex1"/"tex2" above. A named-spectrum
    # string reference (0 floats collected) is silently unsupported, same as
    # before -- no medium in this scene corpus uses one.
    var sa_set = len(params.get_floats("sigma_a")) > 0
    var ss_set = len(params.get_floats("sigma_s")) > 0
    var sa = _psc_get_float_or_rgb(params, "sigma_a", RGB(Float32(0)))
    var ss = _psc_get_float_or_rgb(params, "sigma_s", RGB(Float32(0)))
    var g_val = params.get_float("g", Float32(0))
    var scale = params.get_float("scale", Float32(1))
    # uniformgrid-specific params
    var g_nx = params.get_int("nx", Int32(0))
    var g_ny = params.get_int("ny", Int32(0))
    var g_nz = params.get_int("nz", Int32(0))
    var g_p0 = params.get_rgb("p0", RGB(Float32(0)))
    var g_p1 = params.get_rgb("p1", RGB(Float32(1)))
    var g_density = params.get_floats("density")

    if is_hom:
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_immutable())
        s[0].med_names.append(name_str)
        s[0].med_sa.append(sa.r * scale)
        s[0].med_sa.append(sa.g * scale)
        s[0].med_sa.append(sa.b * scale)
        s[0].med_ss.append(ss.r * scale)
        s[0].med_ss.append(ss.g * scale)
        s[0].med_ss.append(ss.b * scale)
        s[0].med_g.append(g_val)
        s[0].med_grid_idx.append(Int32(-1))
    elif is_grid:
        # PBRT-v4 default for GridMedium sigma_a/sigma_s when unspecified is
        # ConstantSpectrum(1) (see media.cpp) — unlike gonzales's existing
        # HomogeneousMedium path above, which happens to default to 0 (a
        # pre-existing, separate behavior not touched here).
        var sa_eff = sa if sa_set else RGB(Float32(1))
        var ss_eff = ss if ss_set else RGB(Float32(1))
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_immutable())
        s[0].med_names.append(name_str)
        s[0].med_sa.append(sa_eff.r * scale); s[0].med_sa.append(sa_eff.g * scale); s[0].med_sa.append(sa_eff.b * scale)
        s[0].med_ss.append(ss_eff.r * scale); s[0].med_ss.append(ss_eff.g * scale); s[0].med_ss.append(ss_eff.b * scale)
        s[0].med_g.append(g_val)
        s[0].med_grid_idx.append(Int32(len(s[0].grid_nx)))

        s[0].grid_nx.append(g_nx); s[0].grid_ny.append(g_ny); s[0].grid_nz.append(g_nz)
        s[0].grid_p0.append(g_p0.r); s[0].grid_p0.append(g_p0.g); s[0].grid_p0.append(g_p0.b)
        s[0].grid_p1.append(g_p1.r); s[0].grid_p1.append(g_p1.g); s[0].grid_p1.append(g_p1.b)
        for ci in range(16):
            s[0].grid_ctm.append(s[0].ctm[ci])
        s[0].grid_density_base.append(Int32(len(s[0].grid_density)))
        var expected_n = Int(g_nx) * Int(g_ny) * Int(g_nz)
        var copy_n = min(len(g_density), expected_n) if expected_n > 0 else len(g_density)
        for di in range(copy_n):
            s[0].grid_density.append(g_density[di])
        # Pad with zeros if the file had fewer values than nx*ny*nz declared
        # (shouldn't happen for well-formed scenes, but keeps indexing safe).
        for _ in range(copy_n, expected_n):
            s[0].grid_density.append(Float32(0))
    name_buf.free()

def lookup_medium(s: UnsafePointer[SceneParseState, MutExternalOrigin],
                  name: UnsafePointer[UInt8, MutExternalOrigin]) -> Int32:
    if name[0] == UInt8(0):
        return Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=name.as_immutable())
    for i in range(len(s[0].med_names)):
        if s[0].med_names[i] == name_str:
            return Int32(i)
    return Int32(-1)

def handle_medium_interface(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                            s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var inside_buf  = alloc[UInt8](64)
    var outside_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, inside_buf, 64)
    _ = scanner_parse_quoted_string(handle, outside_buf, 64)
    s[0].cur_attr.inside_medium  = lookup_medium(s, inside_buf)
    s[0].cur_attr.outside_medium = lookup_medium(s, outside_buf)
    inside_buf.free(); outside_buf.free()

# ── Shape handlers ────────────────────────────────────────────────────────────

def handle_sphere_shape(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                             s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var params = _psc_collect_params(handle)
    var radius = params.get_float("radius", Float32(1.0))

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

comptime DISK_TESSELLATION_SEGMENTS: Int = 32

def handle_disk_shape(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                           s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    """PBRT `Shape "disk"`: not a native primitive here (unlike sphere) --
    tessellated into a triangle mesh at parse time and handed to the
    existing trianglemesh machinery (BVH, area-light NEE/ReSTIR, GPU, ...)
    via store_mesh, mirroring loopsubdiv's approach above. A disk lies in
    the OBJECT-SPACE z=height plane, an annulus from innerradius to radius
    swept from phi=0 to phimax (degrees, pbrt convention) -- store_mesh
    applies the CTM, so all of this stays in that local frame.

    Tessellated as an EVEN angular fan (innerradius==0, the common case for
    a light fixture) or an even angular strip (innerradius>0) so every
    triangle spans an equal angular wedge and therefore has equal area by
    construction -- this matters because _sample_light_point_and_normal
    (shading.mojo) picks a UNIFORM RANDOM TRIANGLE, not area-weighted, when
    sampling a mesh light; equal-area triangles make that sampling correctly
    uniform-over-surface-area (see project_pbrt_disk_shape_missing /
    project_restir_migration memory for how a non-uniform tessellation
    would silently bias NEE/ReSTIR toward whichever triangles happen to be
    larger). The two triangles WITHIN one annulus strip quad can still
    differ slightly in area from each other (a trapezoid split on the
    diagonal) -- a minor known residual for the innerradius>0 case, not
    worth a fancier non-uniform-strip tessellation to fully equalize.

    Winding is chosen so the un-reversed normal faces OBJECT-SPACE +z,
    matching pbrt's own disk convention; store_mesh's existing
    reverse_orient handling (from `ReverseOrientation`) applies on top,
    exactly as it does for an ordinary "trianglemesh" shape."""
    var params = _psc_collect_params(handle)
    var radius = params.get_float("radius", Float32(1.0))
    var inner_radius = params.get_float("innerradius", Float32(0.0))
    var height = params.get_float("height", Float32(0.0))
    var phimax_deg = params.get_float("phimax", Float32(360.0))
    if phimax_deg > Float32(360.0):
        phimax_deg = Float32(360.0)
    if phimax_deg <= Float32(0.0) or radius <= Float32(0.0) or inner_radius < Float32(0.0) or inner_radius >= radius:
        return

    from std.math import sin as _sin, cos as _cos
    var phimax = phimax_deg * (PI / Float32(180.0))
    comptime n = DISK_TESSELLATION_SEGMENTS

    var pts = List[Float32]()
    var idx = List[Int32]()

    if inner_radius <= Float32(1e-8):
        # Full disk (or pie slice if phimax < 360): a fan from the center.
        pts.append(Float32(0.0)); pts.append(Float32(0.0)); pts.append(height)
        for i in range(n + 1):
            var phi = phimax * Float32(i) / Float32(n)
            pts.append(radius * _cos(phi)); pts.append(radius * _sin(phi)); pts.append(height)
        for i in range(n):
            idx.append(Int32(0)); idx.append(Int32(i + 1)); idx.append(Int32(i + 2))
    else:
        # Annulus: a strip of quads between the inner and outer rings.
        for i in range(n + 1):
            var phi = phimax * Float32(i) / Float32(n)
            pts.append(inner_radius * _cos(phi)); pts.append(inner_radius * _sin(phi)); pts.append(height)
        for i in range(n + 1):
            var phi = phimax * Float32(i) / Float32(n)
            pts.append(radius * _cos(phi)); pts.append(radius * _sin(phi)); pts.append(height)
        var outer0 = Int32(n + 1)
        for i in range(n):
            var in_i = Int32(i); var in_i1 = Int32(i + 1)
            var out_i = outer0 + Int32(i); var out_i1 = outer0 + Int32(i + 1)
            idx.append(in_i); idx.append(out_i); idx.append(out_i1)
            idx.append(in_i); idx.append(out_i1); idx.append(in_i1)

    var n_verts = Int32(len(pts) // 3)
    var n_tris = Int32(len(idx) // 3)
    store_mesh(s, pts.unsafe_ptr(), idx.unsafe_ptr(), n_verts, n_tris)

def handle_shape(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                     s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var shape_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, shape_type, 64)

    var is_tri = _psc_streq(shape_type, "trianglemesh")
    var is_ply = _psc_streq(shape_type, "plymesh")
    var is_curve = _psc_streq(shape_type, "curve")
    var is_sphere = _psc_streq(shape_type, "sphere")
    var is_loopsubdiv = _psc_streq(shape_type, "loopsubdiv")
    var is_disk = _psc_streq(shape_type, "disk")
    shape_type.free()

    if is_disk:
        # Tessellated into a mesh (handle_disk_shape above), so it goes
        # through store_mesh exactly like trianglemesh/loopsubdiv --
        # already usable inside ObjectBegin/ObjectEnd instancing templates,
        # no object_depth restriction needed (unlike curve/sphere above,
        # which are native, uninstanced primitives).
        handle_disk_shape(handle, s)
        return

    if is_curve:
        if s[0].object_depth == 0:
            handle_curve_shape(handle, s)
        else:
            # Curve instancing inside ObjectBegin/ObjectEnd isn't supported yet
            # (only trianglemesh/plymesh templates are, see _psc_finish_object_def) —
            # skip rather than silently adding an un-instanced curve at the
            # template's definition-space position.
            _psc_skip_params(handle)
        return

    if is_sphere:
        if s[0].object_depth == 0:
            handle_sphere_shape(handle, s)
        else:
            # Same rationale as the curve case above — sphere instancing
            # inside ObjectBegin/ObjectEnd isn't supported yet.
            _psc_skip_params(handle)
        return

    if is_loopsubdiv:
        # Task #159: coarse control mesh ("point3 P" + "integer indices") +
        # "integer levels" subdivision count -- see _loopsubdiv_tessellate
        # above. pbrt's own default when "levels" is omitted is 3.
        var ls_params = _psc_collect_params(handle)
        var levels = Int32(ls_params.get_int("levels", 3))
        var p_ctrl = ls_params.take_floats("P")
        var i_ctrl = ls_params.take_ints("indices")
        var n_ctrl_verts = Int32(len(p_ctrl) // 3)
        var n_ctrl_tris = Int32(len(i_ctrl) // 3)
        if n_ctrl_verts <= 0 or n_ctrl_tris <= 0:
            return
        var fin = _loopsubdiv_tessellate(p_ctrl^, i_ctrl^, levels)
        var fin_p = fin[0].copy()
        var fin_i = fin[1].copy()
        var n_verts = Int32(len(fin_p) // 3)
        var n_tris = Int32(len(fin_i) // 3)
        store_mesh(s, fin_p.unsafe_ptr(), fin_i.unsafe_ptr(), n_verts, n_tris)
        return

    if not is_tri and not is_ply:
        _psc_skip_params(handle)
        return

    if is_ply:
        # The only scene-directive param this branch reads is "filename" --
        # the bulk vertex/index/uv/normal data comes from load_ply() parsing
        # a SEPARATE .ply file, not the pbrt token stream, so there's no
        # bulk-array risk going through the generic dictionary here (unlike
        # the trianglemesh "P"/"indices"/"uv" scan below, which stays on its
        # existing dynamic-growth scratch buffers).
        var params = _psc_collect_params(handle)
        var ply_filename_str = params.get_string("filename", "")

        var full_path = alloc[UInt8](PSC_FILE_MAX * 2)
        var dir_len = s[0].scene_dir.byte_length()
        for ki in range(dir_len):
            full_path[ki] = s[0].scene_dir.unsafe_ptr()[ki]
        var fn_bytes = ply_filename_str.unsafe_ptr()
        var fn_len = ply_filename_str.byte_length()
        var fn_i = 0
        while fn_i < fn_len and dir_len + fn_i < PSC_FILE_MAX * 2 - 1:
            full_path[dir_len + fn_i] = fn_bytes[fn_i]
            fn_i += 1
        full_path[dir_len + fn_i] = UInt8(0)

        var ply_pts     = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
        var ply_nv      = alloc[Int32](1)
        var ply_idx     = alloc[UnsafePointer[Int32, MutExternalOrigin]](1)
        var ply_nt      = alloc[Int32](1)
        var ply_uvs     = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
        var ply_has_uvs = alloc[Int32](1)
        var ply_nrm     = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
        var ply_has_nrm = alloc[Int32](1)
        ply_uvs[0] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
        ply_has_uvs[0] = Int32(0)
        ply_nrm[0] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
        ply_has_nrm[0] = Int32(0)
        # For .ply.gz, try the decompressed .ply file first (strip ".gz").
        var fp_len = 0
        while full_path[fp_len] != UInt8(0): fp_len += 1
        var ends_gz = (fp_len >= 4 and
                       full_path[fp_len-3] == UInt8(46) and
                       full_path[fp_len-2] == UInt8(103) and
                       full_path[fp_len-1] == UInt8(122))
        var ok = Int32(0)
        if ends_gz:
            var ap = alloc[UInt8](fp_len - 2)
            for ci in range(fp_len - 3): ap[ci] = full_path[ci]
            ap[fp_len - 3] = UInt8(0)
            ok = load_ply(ap, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
            ap.free()
        if ok == 0:
            ok = load_ply(full_path, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
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

    # take_floats/take_ints move each bulk array's buffer straight out of the
    # dictionary (List.pop, O(1), no copy) -- _psc_collect_params already
    # scanned "P"/"indices"/"uv" directly into each List's own backing
    # buffer sized to exactly what's needed, matching this branch's old
    # hand-rolled scratch-buffer cost with no extra copies. "uv"/"st" are
    # aliases for the same logical param; try "uv" first.
    var params = _psc_collect_params(handle)
    var p_list = params.take_floats("P")
    var i_list = params.take_ints("indices")
    var uv_list = params.take_floats("uv")
    if len(uv_list) == 0:
        uv_list = params.take_floats("st")

    var n_verts = Int32(len(p_list) / 3)
    var n_tris  = Int32(len(i_list) / 3)

    if n_verts <= 0 or n_tris <= 0:
        return

    store_mesh(s, p_list.unsafe_ptr(), i_list.unsafe_ptr(), n_verts, n_tris)
    if Int32(len(uv_list)) >= n_verts * Int32(2):
        for ui in range(Int(n_verts) * 2):
            s[0].meshes[len(s[0].meshes) - 1].uvs.append(uv_list[ui])

# ── Texture handler ───────────────────────────────────────────────────────────

def _psc_get_float_or_rgb(params: ParameterDictionary, name: StringLiteral, default: RGB) -> RGB:
    """"value"/"tex1"/"tex2"-style texture params: a bare float replicates to
    all 3 channels, an rgb triple sets them independently -- same duality as
    material_builder.mojo's "eta" RGB-vs-scalar case, just without a
    named-string third form here."""
    var f = params.get_floats(name)
    if len(f) >= 3:
        return RGB(f[0], f[1], f[2])
    elif len(f) == 1:
        return RGB(f[0])
    return default

def handle_texture(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
                       s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    var tex_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, tex_name, PSC_NAME_MAX)
    var tex_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_type, 64)
    var tex_class = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_class, 64)
    var name_str = String(unsafe_from_utf8_ptr=tex_name.as_immutable())
    tex_name.free()

    if _psc_streq(tex_class, "constant"):
        tex_type.free(); tex_class.free()
        var params = _psc_collect_params(handle)
        var crgb = _psc_get_float_or_rgb(params, "value", RGB(Float32(0.5)))
        s[0].const_tex_names.append(name_str)
        s[0].const_tex_rgb.append(crgb.r)
        s[0].const_tex_rgb.append(crgb.g)
        s[0].const_tex_rgb.append(crgb.b)
        return
    if _psc_streq(tex_class, "checkerboard"):
        tex_type.free(); tex_class.free()
        var params = _psc_collect_params(handle)
        # pbrt defaults: tex1=1 (white), tex2=0 (black), uscale=vscale=1.
        var ktex1 = _psc_get_float_or_rgb(params, "tex1", RGB(Float32(1.0)))
        var ktex2 = _psc_get_float_or_rgb(params, "tex2", RGB(Float32(0.0)))
        var kuscale = params.get_float("uscale", Float32(1.0))
        var kvscale = params.get_float("vscale", Float32(1.0))
        s[0].checker_tex_names.append(name_str)
        s[0].checker_tex1.append(ktex1.r); s[0].checker_tex1.append(ktex1.g); s[0].checker_tex1.append(ktex1.b)
        s[0].checker_tex2.append(ktex2.r); s[0].checker_tex2.append(ktex2.g); s[0].checker_tex2.append(ktex2.b)
        s[0].checker_uscale.append(kuscale)
        s[0].checker_vscale.append(kvscale)
        return
    if not _psc_streq(tex_class, "imagemap"):
        tex_type.free(); tex_class.free()
        _psc_skip_params(handle)
        return

    tex_type.free(); tex_class.free()
    var params = _psc_collect_params(handle)
    var filename = params.get_string("filename", "")
    if filename != "":
        var file_str = s[0].scene_dir + filename
        s[0].tex_names.append(name_str)
        s[0].tex_files.append(file_str)

# ── ObjectBegin/ObjectEnd/ObjectInstance (two-level BVH instancing) ──────────
# See geometry.mojo's Instance_C docs and bvh.mojo's traverse_bvh2_core
# type==6 branch for the traversal side. Design: geometry inside
# ObjectBegin/ObjectEnd is parsed normally (baked at whatever CTM is active
# during that block, "definition space") but tagged is_object_template=True so
# finalize_scene excludes it from the ordinary top-level primitive list —
# instead a private BLAS is built once per template, and each ObjectInstance
# placement contributes a small TLAS leaf (transform + BLAS reference) rather
# than a duplicated copy of the geometry.

def _psc_finish_object_def(s: UnsafePointer[SceneParseState, MutExternalOrigin]):
    """Called when the outermost ObjectEnd closes a template: mark the
    meshes captured since the matching ObjectBegin as template-only and
    record the template's (name, mesh range, definition-time CTM) for
    finalize_scene and later ObjectInstance directives."""
    var start = Int(s[0].pending_object_start)
    var end   = len(s[0].meshes)
    if end <= start:
        return  # empty object (e.g. only unsupported AreaLightSource/curves) — nothing to instance
    for i in range(start, end):
        s[0].meshes[i].is_object_template = True
    s[0].object_names.append(s[0].pending_object_name)
    s[0].object_mesh_start.append(Int32(start))
    s[0].object_mesh_end.append(Int32(end))
    for ci in range(16):
        s[0].object_ctm.append(s[0].pending_object_ctm[ci])

def _psc_emit_object_instance(s: UnsafePointer[SceneParseState, MutExternalOrigin], name: String):
    """Called on ObjectInstance "name": look up the named template and record
    a placement (template index + obj_to_world/world_to_obj transforms,
    derived from the CTM active now vs. the CTM active at that template's
    ObjectBegin) for finalize_scene to turn into an Instance_C."""
    var tmpl_idx = -1
    for i in range(len(s[0].object_names)):
        if s[0].object_names[i] == name:
            tmpl_idx = i
            break
    if tmpl_idx < 0:
        return  # unknown/empty object name — nothing to place (e.g. all-skipped-content object)

    # obj_to_world = CTM_now * inverse(CTM_at_ObjectBegin) — since template
    # geometry is already baked in "CTM_at_ObjectBegin space", this maps it
    # into this placement's world position without re-parsing/duplicating it.
    var mdef = alloc[Float32](16)
    for ci in range(16): mdef[ci] = s[0].object_ctm[tmpl_idx * 16 + ci]
    var mdef_inv = alloc[Float32](16)
    _ = matrix_invert(mdef, mdef_inv)
    var obj_to_world = alloc[Float32](16)
    matrix_multiply(s[0].ctm.unsafe_ptr(), mdef_inv, obj_to_world)
    var world_to_obj = alloc[Float32](16)
    _ = matrix_invert(obj_to_world, world_to_obj)

    s[0].instance_template_idx.append(Int32(tmpl_idx))
    for ci in range(16):
        s[0].instance_obj_to_world.append(obj_to_world[ci])
        s[0].instance_world_to_obj.append(world_to_obj[ci])

    mdef.free(); mdef_inv.free(); obj_to_world.free(); world_to_obj.free()

# ── Main parse loop ───────────────────────────────────────────────────────────

def parse_scene_file(handle: UnsafePointer[PbrtScanner, MutExternalOrigin],
              s: UnsafePointer[SceneParseState, MutExternalOrigin]):
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
            if s[0].object_depth == 0:
                s[0].pending_object_name  = String(unsafe_from_utf8_ptr=obj_name.as_immutable())
                s[0].pending_object_start = Int32(len(s[0].meshes))
                s[0].pending_object_ctm   = s[0].ctm.copy()
            obj_name.free()
            s[0].object_depth += 1
        elif _psc_streq(kw_buf, "ObjectEnd"):
            if s[0].object_depth > 0:
                s[0].object_depth -= 1
                if s[0].object_depth == 0:
                    _psc_finish_object_def(s)
        elif _psc_streq(kw_buf, "ObjectInstance"):
            var obj_name = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            var inst_name = String(unsafe_from_utf8_ptr=obj_name.as_immutable())
            obj_name.free()
            _psc_emit_object_instance(s, inst_name)
        elif _psc_streq(kw_buf, "Shape"):
            # Shapes inside an ObjectBegin/ObjectEnd block ARE parsed (into a
            # template mesh range, see _psc_finish_object_def) — unlike
            # AreaLightSource/LightSource below, which stay scoped out.
            handle_shape(handle, s)
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

            # `.pbrt.gz` includes (e.g. pbrt-v4-scenes' curve/hair geometry)
            # are NOT decompressed by the tokenizer -- without this, the
            # scanner opens the raw gzip bytes as if they were text and
            # silently tokenizes garbage, producing zero shapes with no
            # error at all. Auto-decompress once into a cached ".pbrt"
            # sibling next to the source (mirrors the pre-existing ".ply.gz"
            # sibling-file convention above), then open that instead.
            var open_path: UnsafePointer[UInt8, MutExternalOrigin] = inc_path
            var inc_path_len = dlen + fi
            var ends_gz = (inc_path_len >= 3 and
                           inc_path[inc_path_len-3] == UInt8(46) and
                           inc_path[inc_path_len-2] == UInt8(103) and
                           inc_path[inc_path_len-1] == UInt8(122))
            var stripped = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling()
            if ends_gz:
                stripped = alloc[UInt8](inc_path_len - 2)
                for ci in range(inc_path_len - 3):
                    stripped[ci] = inc_path[ci]
                stripped[inc_path_len - 3] = UInt8(0)
                var stripped_str = String(unsafe_from_utf8_ptr=stripped.as_immutable())
                if not exists(stripped_str):
                    var inc_path_str = String(unsafe_from_utf8_ptr=inc_path.as_immutable())
                    try:
                        _ = run("gzip -dk '" + inc_path_str + "'")
                    except:
                        pass
                open_path = stripped

            var sub_handle = scanner_open(open_path)
            if scanner_is_at_end(sub_handle) == 0:
                # Splice the included file's bytes in front of this scanner's
                # remaining bytes so both halves form one continuous token
                # stream, matching pbrt's own tokenizer semantics. This lets a
                # directive's parameter list (e.g. MakeNamedMedium) continue
                # across the Include boundary instead of being cut short.
                var inc_len = Int(sub_handle[0].total_bytes)
                var rest_start = Int(handle[0].cursor)
                var rest_len = Int(handle[0].total_bytes) - rest_start
                var merged_len = inc_len + rest_len
                var merged = alloc[UInt8](merged_len + 1)
                for mi in range(inc_len):
                    merged[mi] = sub_handle[0].buffer[mi]
                for mi in range(rest_len):
                    merged[inc_len + mi] = handle[0].buffer[rest_start + mi]
                merged[merged_len] = UInt8(0)
                if _is_real_ptr(handle[0].buffer):
                    handle[0].buffer.free()
                handle[0].buffer = merged
                handle[0].total_bytes = Int32(merged_len)
                handle[0].cursor = Int32(0)
                handle[0].is_at_end = Int32(0)
            else:
                var inc_str = String(unsafe_from_utf8_ptr=inc_name.as_immutable())
                if inc_str.endswith(".xz"):
                    print("Warning: cannot open include (decompress first with xz -dk):", inc_str)
                elif inc_str.endswith(".gz"):
                    print("Warning: cannot open include (gzip decompression failed):", inc_str)
                else:
                    print("Warning: cannot open include:", inc_str)
            scanner_free(sub_handle)
            if ends_gz:
                stripped.free()
            inc_name.free(); inc_path.free()
        elif _psc_streq(kw_buf, "Material"):
            _psc_handle_make_named_material(handle, s, True)
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
                         dst: UnsafePointer[Float32, MutExternalOrigin]):
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
                               dst: UnsafePointer[Float32, MutExternalOrigin]):
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

comptime CURVE_GROUP_MAX: Int = 2   # hard cap on pieces merged per BVH leaf — see _curve_greedy_groups

# ── Curve BVH-leaf grouping ─────────────────────────────────────────────────
# Curly curves are chopped into up to CURVE_N_PIECES locally-linear pieces
# (see the flatness test in finalize_scene). Giving every piece its own BVH
# leaf fixed a severe GPU divergence/false-positive-candidate problem from
# one loose whole-segment leaf, but made BVH construction ~7x slower on
# curly-heavy scenes (more leaves to build). Curl is usually gradual rather
# than zigzag, so adjacent pieces of the same curly curve are very often
# still collinear with each other even when the whole 4-control-point
# segment isn't — this greedily merges such runs into one leaf, so a curl
# typically costs 2-4 leaves instead of always exactly CURVE_N_PIECES.

def _curve_greedy_groups(
    curve: Curve_C,
    n_pieces: Int,
    out_first: UnsafePointer[Int32, MutExternalOrigin],
    out_count: UnsafePointer[Int32, MutExternalOrigin],
    write: Bool,
) -> Int:
    """Greedy-merge adjacent pieces of one curve into flat runs. Returns the
    number of groups. If write=True, fills out_first/out_count (each must
    have capacity >= n_pieces) with (first_piece, piece_count) per group; if
    write=False the out pointers are ignored — used for a cheap first pass
    to size the final arrays before allocating them."""
    var pts = InlineArray[Vec3f, CURVE_N_PIECES + 1](fill=Vec3f(0, 0, 0))
    for k in range(n_pieces + 1):
        pts[k] = curve_bspline_point(curve, Float32(k) / Float32(n_pieces))
    var maxw = max(curve.width0, curve.width1)
    var thresh = maxw * Float32(0.5)

    var n_groups = 0
    var start = 0
    while start < n_pieces:
        var end = start + 1
        while end < n_pieces and end - start < CURVE_GROUP_MAX:
            # Would including piece `end` (run becomes [start, end+1)) still
            # look flat? Same chord-deviation test as the whole-segment
            # flatness check, just applied to this candidate sub-run.
            # Capped at CURVE_GROUP_MAX regardless of flatness: the
            # flatness-only version measured avg ~4 pieces/group, which
            # brought back most of the divergent-internal-loop cost that
            # per-piece leaves were meant to remove (render time regressed
            # 2.2s -> 14.9s on furball even though BVH build got faster) —
            # the cap bounds the worst case while still merging the common,
            # genuinely-flat 2-piece case.
            var chord = pts[end + 1] - pts[start]
            var chord_len_sq = dot(chord, chord)
            var ok = True
            if chord_len_sq > Float32(1e-16):
                var inv_len = Float32(1.0) / sqrt(chord_len_sq)
                var dir = chord * inv_len
                for k in range(start + 1, end + 1):
                    var v = pts[k] - pts[start]
                    var proj = dot(v, dir)
                    var perp = v - proj * dir
                    if sqrt(dot(perp, perp)) > thresh:
                        ok = False
                        break
            if not ok:
                break
            end += 1
        if write:
            out_first[n_groups] = Int32(start)
            out_count[n_groups] = Int32(end - start)
        n_groups += 1
        start = end
    return n_groups

# ── Scene finalization ────────────────────────────────────────────────────────

def finalize_scene(s: UnsafePointer[SceneParseState, MutExternalOrigin],
                 psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
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
    # A Shape parsed with no active Material directive keeps cur_attr.mat_idx at
    # its -1 "unset" sentinel (parse_types.mojo's SceneParseState init). PBRT's
    # own default in that case is a plain 50%-grey diffuse material (see
    # DiffuseMaterial::Create's `reflectance` fallback) — NOT whatever happens to
    # be the first Material statement encountered in the file. Previously -1 was
    # clamped to material index 0 below, so an unrelated shape silently borrowed
    # material index 0's appearance (e.g. ganesha.pbrt's floor, parsed before the
    # scene's sole "coateddiffuse" Material statement, rendered with Ganesha's
    # near-mirror coat instead of a matte grey floor). Only append the synthetic
    # default when something actually needs it.
    var needs_default_mat = False
    for i in range(len(s[0].meshes)):
        if s[0].meshes[i].mat_idx == Int32(-1) and not s[0].meshes[i].is_area_light:
            needs_default_mat = True
            break
    if not needs_default_mat:
        for i in range(len(s[0].spheres_mat)):
            if s[0].spheres_mat[i] == Int32(-1):
                needs_default_mat = True
                break
    if not needs_default_mat:
        for i in range(len(s[0].curves_mat)):
            var curve_is_al = i < len(s[0].curves_al) and s[0].curves_al[i]
            if s[0].curves_mat[i] == Int32(-1) and not curve_is_al:
                needs_default_mat = True
                break
    var default_mat_idx = Int32(-1)
    if needs_default_mat:
        var default_nm = NamedMaterial(String("__default_diffuse__"))
        default_nm.kind = MatKind.diffuse
        default_nm.albedo = RGB(Float32(0.5), Float32(0.5), Float32(0.5))
        s[0].named_materials.append(default_nm^)
        default_mat_idx = Int32(len(s[0].named_materials) - 1)

    var n_regular = len(s[0].named_materials)

    var n_al_mesh = 0
    for i in range(len(s[0].meshes)):
        if s[0].meshes[i].is_area_light:
            n_al_mesh += 1

    # Curve area lights (see curves_al/curves_al_rgb) get their own synthetic
    # material slots too, right after the mesh area-light slots — one per
    # emissive curve *segment* (a `Shape "curve"` directive splits into
    # several local B-spline segments, see curves_mat; they all share the
    # same cur_attr.al_rgb at parse time, so this is simply the simplest
    # correct granularity, not a meaningful dedup opportunity lost).
    var n_al_curve = 0
    for i in range(len(s[0].curves_al)):
        if s[0].curves_al[i]:
            n_al_curve += 1

    var n_al = n_al_mesh + n_al_curve
    var n_mats = n_regular + n_al

    # "measured" materials: dedup by resolved .bsdf path, load each unique
    # file once via the real tensor-file loader (measured_bsdf.mojo). Stage 1
    # of the real-MeasuredBxDF port — shading doesn't consume this yet
    # (material_builder.mojo still routes "measured" through the approximate
    # conductor path); this just populates measured_brdfs/measured_idx so
    # Stage 2 can flip the switch without further parser changes.
    var measured_paths = List[String]()
    var measured_ok    = List[Bool]()
    var measured_list  = List[MeasuredBRDF_C]()
    for i in range(n_regular):
        var mpath = s[0].named_materials[i].measured_bsdf_path
        if mpath == "":
            continue
        var already = False
        for j in range(len(measured_paths)):
            if measured_paths[j] == mpath:
                already = True
                break
        if already:
            continue
        var (mok, mb) = load_measured_brdf_full(mpath)
        if not mok:
            print("Warning: could not load full measured BRDF '" + mpath + "' (real MeasuredBxDF unavailable, using approximation)")
        measured_paths.append(mpath)
        measured_ok.append(mok)
        measured_list.append(mb)
    psc[0].measured_count = Int32(len(measured_list))
    var measured_brdfs_buf = alloc[MeasuredBRDF_C](max(len(measured_list), 1))
    for i in range(len(measured_list)):
        measured_brdfs_buf[i] = measured_list[i]
    psc[0].measured_brdfs = measured_brdfs_buf

    var mats = alloc[Material_C](max(n_mats, 1))
    for i in range(n_regular):
        var nm3 = s[0].named_materials[i]
        var material_kind = nm3.kind
        var ior = nm3.ior
        mats[i].type = material_kind
        mats[i].tex_idx = nm3.tex_idx
        mats[i].roughU  = nm3.roughness_u
        mats[i].roughV  = nm3.roughness_v
        mats[i].normal_tex_idx = nm3.normal_tex_idx
        mats[i].rough_tex_idx = nm3.rough_tex_idx
        mats[i].medium_interface_idx = Int32(-1)
        if nm3.measured_bsdf_path == "":
            mats[i].measured_idx = Int32(-1)
        else:
            var midx = Int32(-1)
            for j in range(len(measured_paths)):
                if measured_paths[j] == nm3.measured_bsdf_path:
                    if measured_ok[j]:
                        midx = Int32(j)
                    break
            mats[i].measured_idx = midx
            if midx >= Int32(0):
                # Real MeasuredBxDF loaded successfully -- route through the
                # real algorithm (shading.mojo's shade_measured) instead of
                # material_builder.mojo's achromatic rough-conductor
                # approximation (material_kind, still MatKind.conductor,
                # stays as the graceful fallback for a load failure above).
                mats[i].type = MatKind.measured
        mats[i].checker_tex1   = nm3.checker_tex1
        mats[i].checker_tex2   = nm3.checker_tex2
        mats[i].checker_uscale = nm3.checker_uscale
        mats[i].checker_vscale = nm3.checker_vscale
        if material_kind == MatKind.dielectric:
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0))
        elif material_kind == MatKind.coated_diffuse:
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif material_kind == MatKind.coated_conductor:
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif material_kind == MatKind.thin_dielectric:
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0))
        elif material_kind == MatKind.mix:
            var idx1 = Int32(0)
            var idx2 = Int32(0)
            for j in range(n_regular):
                if s[0].named_materials[j].name == nm3.mix_name1: idx1 = Int32(j)
                if s[0].named_materials[j].name == nm3.mix_name2: idx2 = Int32(j)
            mats[i].tex_idx = (idx2 << 16) | (idx1 & Int32(0xFFFF))
            mats[i].roughU  = nm3.mix_amount
            mats[i].albedo  = nm3.albedo
            mats[i].emission = RGB(Float32(0))
        elif material_kind == MatKind.diffuse_transmit:
            mats[i].albedo = nm3.albedo
            mats[i].emission = nm3.transmittance
        elif material_kind == MatKind.hair:
            mats[i].albedo = nm3.albedo   # sigma_a per channel (absorption coefficient)
            mats[i].emission = RGB(Float32(1.55), Float32(0), Float32(0))  # IOR for hair cuticle
            if nm3.roughness_u == Float32(0):
                mats[i].roughU = Float32(0.3)  # betaM default
            else:
                mats[i].roughU = nm3.roughness_u
            if nm3.roughness_v == Float32(0):
                mats[i].roughV = Float32(0.3)  # betaN default
            else:
                mats[i].roughV = nm3.roughness_v
            mats[i].type = MatKind.hair
        else:
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(Float32(0))

    # ---- Meshes + area lights ----
    var n_meshes = len(s[0].meshes)
    var meshes   = alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts  = alloc[UnsafePointer[Float32, MutExternalOrigin]](max(n_meshes, 1))
    var out_vis  = alloc[UnsafePointer[Int64, MutExternalOrigin]](max(n_meshes, 1))
    var out_fis  = alloc[UnsafePointer[Int64, MutExternalOrigin]](max(n_meshes, 1))
    var out_nv    = alloc[Int32](max(n_meshes, 1))
    var out_nt    = alloc[Int32](max(n_meshes, 1))
    var out_uv_nv = alloc[Int32](max(n_meshes, 1))
    var out_nrm_nv = alloc[Int32](max(n_meshes, 1))

    # al_list (used for NEE light sampling) covers mesh area lights (kind=0,
    # filled below) AND curve area lights (kind=1, appended once curve_buf
    # is built further down in "Native curves") — sized for both up front
    # since it's one contiguous allocation.
    var al_list  = alloc[AreaLight_C](max(n_al_mesh + n_al_curve, 1))
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
            meshes[i].uvs = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
            out_uv_nv[i] = Int32(0)
        if len(ma.normals) >= nv * 3:
            var nrm_c = alloc[Float32](nv * 3)
            for ni in range(nv * 3): nrm_c[ni] = ma.normals[ni]
            meshes[i].normals = nrm_c
            out_nrm_nv[i] = Int32(nv)
        else:
            meshes[i].normals = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
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
            al_list[al_idx].kind       = Int8(0)
            mats[al_mat_base + al_idx].type     = Int8(2)
            mats[al_mat_base + al_idx].albedo   = RGB(Float32(0))
            mats[al_mat_base + al_idx].emission = em
            mats[al_mat_base + al_idx].tex_idx  = Int32(-1)
            mats[al_mat_base + al_idx].roughU   = Float32(0)
            mats[al_mat_base + al_idx].roughV   = Float32(0)
            mats[al_mat_base + al_idx].normal_tex_idx = Int32(-1)
            mats[al_mat_base + al_idx].rough_tex_idx = Int32(-1)
            mats[al_mat_base + al_idx].medium_interface_idx = Int32(-1)
            mats[al_mat_base + al_idx].measured_idx = Int32(-1)
            al_count += 1

    # ---- Curve area light material slots ----
    # Each emissive curve segment gets its own synthetic material slot here
    # (used by the direct-hit path in shade_nee_core), same as before. Their
    # AreaLight_C/NEE entries (al_list[n_al_mesh:]) are appended further down
    # in "Native curves" once curve_buf/curve_n_pieces exist — total_area
    # needs the curve's actual piece tessellation (curve_light_tube_area).
    var curve_al_mat_idx = alloc[Int32](max(len(s[0].curves_al), 1))
    var curve_al_running = Int32(0)
    for ci in range(len(s[0].curves_al)):
        if s[0].curves_al[ci]:
            var slot = al_mat_base + n_al_mesh + Int(curve_al_running)
            curve_al_mat_idx[ci] = Int32(slot)
            var em = s[0].curves_al_rgb[ci]
            mats[slot].type     = MatKind.area_light
            mats[slot].albedo   = RGB(Float32(0))
            mats[slot].emission = em
            mats[slot].tex_idx  = Int32(-1)
            mats[slot].roughU   = Float32(0)
            mats[slot].roughV   = Float32(0)
            mats[slot].normal_tex_idx = Int32(-1)
            mats[slot].rough_tex_idx = Int32(-1)
            mats[slot].medium_interface_idx = Int32(-1)
            mats[slot].measured_idx = Int32(-1)
            curve_al_running += 1
        else:
            curve_al_mat_idx[ci] = Int32(-1)

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
        psc[0].medium_ifaces = UnsafePointer[MediumInterface_C, MutExternalOrigin].unsafe_dangling()
        psc[0].medium_iface_count = Int32(0)

    var total_tris = Int32(0)
    for i in range(n_meshes):
        if not s[0].meshes[i].is_object_template:
            total_tris += Int32(len(s[0].meshes[i].face_idxs))

    # Native curves: precompute per-curve piece count via the same flatness
    # test used for the Curve_C upload below, then greedily merge adjacent
    # flat-enough pieces of each curly curve into single BVH leaves (see
    # _curve_greedy_groups) so the leaf count stays close to the number of
    # visually-distinct bends, not always CURVE_N_PIECES.
    var n_curves = Int32(len(s[0].curves_mat))
    var curve_n_pieces = alloc[Int32](max(Int(n_curves), 1))
    var curve_group_base = alloc[Int32](max(Int(n_curves), 1))
    var total_curve_groups = Int32(0)
    # Never dereferenced (the counting pass below only counts groups; the
    # write=False branch of _curve_greedy_groups skips all writes) — just
    # needs to be a valid, non-dangling pointer to satisfy the signature.
    var count_pass_scratch = alloc[Int32](1)
    for i in range(Int(n_curves)):
        var cb = i * 12
        var cx0 = s[0].curves_cp[cb+0]; var cy0 = s[0].curves_cp[cb+1]; var cz0 = s[0].curves_cp[cb+2]
        var cx3 = s[0].curves_cp[cb+9]; var cy3 = s[0].curves_cp[cb+10]; var cz3 = s[0].curves_cp[cb+11]
        var chord_x = cx3 - cx0; var chord_y = cy3 - cy0; var chord_z = cz3 - cz0
        var chord_len = sqrt(chord_x*chord_x + chord_y*chord_y + chord_z*chord_z)
        var max_dev = Float32(0.0)
        if chord_len > Float32(1e-8):
            var dcx = chord_x / chord_len; var dcy = chord_y / chord_len; var dcz = chord_z / chord_len
            for k in range(1, 3):
                var vx = s[0].curves_cp[cb+k*3+0] - cx0
                var vy = s[0].curves_cp[cb+k*3+1] - cy0
                var vz = s[0].curves_cp[cb+k*3+2] - cz0
                var ccx = vy*dcz - vz*dcy
                var ccy = vz*dcx - vx*dcz
                var ccz = vx*dcy - vy*dcx
                var dist = sqrt(ccx*ccx + ccy*ccy + ccz*ccz)
                if dist > max_dev: max_dev = dist
        var maxw = max(s[0].curves_w0[i], s[0].curves_w1[i])
        curve_n_pieces[i] = Int32(1) if max_dev < maxw * Float32(0.5) else Int32(CURVE_N_PIECES)

        var curve_i = Curve_C(
            Point3f(s[0].curves_cp[cb+0], s[0].curves_cp[cb+1], s[0].curves_cp[cb+2]),
            Point3f(s[0].curves_cp[cb+3], s[0].curves_cp[cb+4], s[0].curves_cp[cb+5]),
            Point3f(s[0].curves_cp[cb+6], s[0].curves_cp[cb+7], s[0].curves_cp[cb+8]),
            Point3f(s[0].curves_cp[cb+9], s[0].curves_cp[cb+10], s[0].curves_cp[cb+11]),
            s[0].curves_w0[i], s[0].curves_w1[i], s[0].curves_mat[i], curve_n_pieces[i])
        var ngroups = _curve_greedy_groups(curve_i, Int(curve_n_pieces[i]), count_pass_scratch, count_pass_scratch, False)
        curve_group_base[i] = total_curve_groups
        total_curve_groups += Int32(ngroups)

    var total_instances = Int32(len(s[0].instance_template_idx))
    var total_prims_gpu = total_tris + total_curve_groups
    var total_prims = total_prims_gpu + total_instances

    var prim_bounds = alloc[Float32](Int(total_prims) * 6)
    var tri_mesh    = alloc[Int32](max(Int(total_tris), 1))
    var tri_local   = alloc[Int32](max(Int(total_tris), 1))

    var flat_idx = Int32(0)
    for mi in range(n_meshes):
        if s[0].meshes[mi].is_object_template:
            continue
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

    # Native curve groups: one BVH leaf per greedily-merged run of pieces
    # (see _curve_greedy_groups), each with a tight AABB — the union of that
    # run's individual piece bounds, still far tighter than the old
    # whole-segment hull since a run rarely spans the entire curly curve.
    var group_curve_idx = alloc[Int32](max(Int(total_curve_groups), 1))
    var group_id2 = alloc[Int32](max(Int(total_curve_groups), 1))  # packed first_piece*8 + piece_count
    var group_first_scratch = alloc[Int32](CURVE_N_PIECES)
    var group_count_scratch = alloc[Int32](CURVE_N_PIECES)
    for ci in range(Int(n_curves)):
        var base = ci * 12
        var curve_i = Curve_C(
            Point3f(s[0].curves_cp[base+0], s[0].curves_cp[base+1], s[0].curves_cp[base+2]),
            Point3f(s[0].curves_cp[base+3], s[0].curves_cp[base+4], s[0].curves_cp[base+5]),
            Point3f(s[0].curves_cp[base+6], s[0].curves_cp[base+7], s[0].curves_cp[base+8]),
            Point3f(s[0].curves_cp[base+9], s[0].curves_cp[base+10], s[0].curves_cp[base+11]),
            s[0].curves_w0[ci], s[0].curves_w1[ci], s[0].curves_mat[ci], curve_n_pieces[ci])
        var ngroups = _curve_greedy_groups(curve_i, Int(curve_n_pieces[ci]), group_first_scratch, group_count_scratch, True)
        for g in range(ngroups):
            var global_group = Int(curve_group_base[ci]) + g
            var first_piece = Int(group_first_scratch[g])
            var piece_count = Int(group_count_scratch[g])
            group_curve_idx[global_group] = Int32(ci)
            group_id2[global_group] = Int32(first_piece * 8 + piece_count)
            var (xmin, ymin, zmin, xmax, ymax, zmax) = curve_piece_bounds(curve_i, first_piece)
            for p in range(first_piece + 1, first_piece + piece_count):
                var (pxmin, pymin, pzmin, pxmax, pymax, pzmax) = curve_piece_bounds(curve_i, p)
                xmin = min(xmin, pxmin); ymin = min(ymin, pymin); zmin = min(zmin, pzmin)
                xmax = max(xmax, pxmax); ymax = max(ymax, pymax); zmax = max(zmax, pzmax)
            var b = (Int(total_tris) + global_group) * 6
            prim_bounds[b+0] = xmin; prim_bounds[b+1] = ymin; prim_bounds[b+2] = zmin
            prim_bounds[b+3] = xmax; prim_bounds[b+4] = ymax; prim_bounds[b+5] = zmax
    group_first_scratch.free(); group_count_scratch.free(); count_pass_scratch.free()

    # ---- Object instancing: one BLAS per template, then a TLAS instance leaf
    # (transform + BLAS reference) per ObjectInstance placement — see
    # geometry.mojo's Instance_C docs. A BLAS is a private BVH2 built over
    # just that template's own mesh range, with ordinary type==0 PrimId_C
    # entries referencing the SAME GLOBAL `meshes` array (no per-BLAS mesh
    # storage, no geometry duplication).
    var n_templates = len(s[0].object_names)
    var blas_nodes_arr   = alloc[UnsafePointer[BVH2Node, MutExternalOrigin]](max(n_templates, 1))
    var blas_primids_arr = alloc[UnsafePointer[PrimId_C, MutExternalOrigin]](max(n_templates, 1))
    # Per-BLAS array lengths — the CPU traversal side never needs these (it
    # just walks from node/primid index 0, self-describing via each node's
    # offset/count), but GPU upload does: it copies each BLAS's arrays into
    # their own device buffers and needs to know how many bytes that is.
    var blas_node_counts   = alloc[Int32](max(n_templates, 1))
    var blas_primid_counts = alloc[Int32](max(n_templates, 1))
    var template_mesh_start = alloc[Int32](max(n_templates, 1))
    var template_mesh_end   = alloc[Int32](max(n_templates, 1))
    for tmpl in range(n_templates):
        var mstart = Int(s[0].object_mesh_start[tmpl])
        var mend   = Int(s[0].object_mesh_end[tmpl])
        template_mesh_start[tmpl] = Int32(mstart)
        template_mesh_end[tmpl]   = Int32(mend)
        var t_tris = Int32(0)
        for mi in range(mstart, mend):
            t_tris += Int32(len(s[0].meshes[mi].face_idxs))
        var t_bounds = alloc[Float32](max(Int(t_tris), 1) * 6)
        var t_mesh   = alloc[Int32](max(Int(t_tris), 1))
        var t_local  = alloc[Int32](max(Int(t_tris), 1))
        var t_flat = Int32(0)
        for mi in range(mstart, mend):
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
                var tb = Int(t_flat) * 6
                t_bounds[tb+0] = min(x0, min(x1, x2))
                t_bounds[tb+1] = min(y0, min(y1, y2))
                t_bounds[tb+2] = min(z0, min(z1, z2))
                t_bounds[tb+3] = max(x0, max(x1, x2))
                t_bounds[tb+4] = max(y0, max(y1, y2))
                t_bounds[tb+5] = max(z0, max(z1, z2))
                t_mesh[Int(t_flat)]  = Int32(mi)
                t_local[Int(t_flat)] = Int32(ti)
                t_flat += 1
        var t_max_nodes = max(Int(t_tris) * 2 + 4, 1)
        var t_nodes = alloc[BVH2Node](t_max_nodes)
        var t_order = alloc[Int32](max(Int(t_tris), 1))
        var t_node_count = build_bvh2(t_bounds, t_tris, t_nodes, t_order)
        t_bounds.free()
        var t_prim_ids = alloc[PrimId_C](max(Int(t_tris), 1))
        for k in range(Int(t_tris)):
            var orig = Int(t_order[k])
            var mi = Int(t_mesh[orig])
            var ti = Int(t_local[orig])
            var mat_idx = Int(s[0].meshes[mi].mat_idx)
            var mat_idx_r = mat_idx if mat_idx >= 0 else Int(default_mat_idx)
            t_prim_ids[k] = PrimId_C(Int64(mi), Int64(ti * 3), Int64(mat_idx_r), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        t_mesh.free(); t_local.free(); t_order.free()
        blas_nodes_arr[tmpl]   = t_nodes
        blas_primids_arr[tmpl] = t_prim_ids
        blas_node_counts[tmpl]   = t_node_count
        blas_primid_counts[tmpl] = t_tris

    var instances_c = alloc[Instance_C](max(Int(total_instances), 1))
    for k in range(Int(total_instances)):
        var tmpl_idx = Int(s[0].instance_template_idx[k])
        var o2w = SIMD[DType.float32, 16](0.0)
        var w2o = SIMD[DType.float32, 16](0.0)
        for ci in range(16):
            o2w[ci] = s[0].instance_obj_to_world[k*16+ci]
            w2o[ci] = s[0].instance_world_to_obj[k*16+ci]
        instances_c[k] = Instance_C(o2w, w2o, Int32(tmpl_idx))

        # World-space AABB for the TLAS leaf: transform the BLAS root's 8 corners.
        var root = blas_nodes_arr[tmpl_idx][0]
        var wxmin = Float32(1e38); var wymin = Float32(1e38); var wzmin = Float32(1e38)
        var wxmax = Float32(-1e38); var wymax = Float32(-1e38); var wzmax = Float32(-1e38)
        for corner in range(8):
            var cx = root.max.x if (corner & 1) != 0 else root.min.x
            var cy = root.max.y if (corner & 2) != 0 else root.min.y
            var cz = root.max.z if (corner & 4) != 0 else root.min.z
            var wx = o2w[0]*cx + o2w[4]*cy + o2w[8]*cz  + o2w[12]
            var wy = o2w[1]*cx + o2w[5]*cy + o2w[9]*cz  + o2w[13]
            var wz = o2w[2]*cx + o2w[6]*cy + o2w[10]*cz + o2w[14]
            wxmin = min(wxmin, wx); wymin = min(wymin, wy); wzmin = min(wzmin, wz)
            wxmax = max(wxmax, wx); wymax = max(wymax, wy); wzmax = max(wzmax, wz)
        var ib = (Int(total_tris) + Int(total_curve_groups) + k) * 6
        prim_bounds[ib+0] = wxmin; prim_bounds[ib+1] = wymin; prim_bounds[ib+2] = wzmin
        prim_bounds[ib+3] = wxmax; prim_bounds[ib+4] = wymax; prim_bounds[ib+5] = wzmax

    # ---- GPU-safe TLAS: tris + curves only ----
    # GPU's own device-side scene upload (gpu_upload_scene, via
    # _gpu_upload_scene in pipeline.mojo) reads psc[0].bvh_nodes/prim_ids
    # directly and has no BLAS/instance buffers at all. Build it over just
    # the first `total_prims_gpu` entries of `prim_bounds` (the instance AABBs
    # live in the tail, past this count, and are simply never read here) so
    # no PrimId_C.type==6 leaf can ever appear in GPU's uploaded arrays.
    #
    # This used to be the ONLY top-level BVH build, shared by CPU and GPU —
    # splitting it in two was necessary after testing showed GPU crashing
    # (CUDA_ERROR_ILLEGAL_ADDRESS) on a scene with real instances, even with a
    # defensive dangling-pointer guard in the shared traversal code guarding
    # the type==6 branch (bisected: removing that branch entirely made the
    # crash disappear, so some GPU-codegen quirk with the guard's own pointer
    # comparison was still letting a dangling dereference through — not worth
    # chasing further when structurally preventing GPU from ever seeing one
    # of these leaves is the clean fix anyway).
    var max_bvh_nodes_gpu = Int(total_prims_gpu) * 2 + 4
    var bvh_nodes_gpu = alloc[BVH2Node](max_bvh_nodes_gpu)
    var bvh_order_gpu = alloc[Int32](Int(total_prims_gpu))
    var node_count_gpu = build_bvh2(prim_bounds, total_prims_gpu, bvh_nodes_gpu, bvh_order_gpu)

    # ---- CPU-inclusive TLAS: tris + curves + instances ----
    var max_bvh_nodes = Int(total_prims) * 2 + 4
    var bvh_nodes = alloc[BVH2Node](max_bvh_nodes)
    var bvh_order = alloc[Int32](Int(total_prims))
    var node_count = build_bvh2(prim_bounds, total_prims, bvh_nodes, bvh_order)

    prim_bounds.free()

    var prim_ids_gpu = alloc[PrimId_C](Int(total_prims_gpu))
    var prim_ids = alloc[PrimId_C](Int(total_prims))

    var mesh_al_idx = alloc[Int32](max(n_meshes, 1))
    var running_al = Int32(0)
    for mi in range(n_meshes):
        if s[0].meshes[mi].is_area_light:
            mesh_al_idx[mi] = running_al
            running_al += 1
        else:
            mesh_al_idx[mi] = Int32(-1)

    # GPU-safe PrimId assignment — tris + curves only (no instance branch;
    # `orig` here is always < total_tris + total_curve_groups). Same
    # tri/curve logic as the CPU-inclusive loop below.
    for k in range(Int(total_prims_gpu)):
        var orig = Int(bvh_order_gpu[k])
        if orig < Int(total_tris):
            var mi = Int(tri_mesh[orig])
            var ti = Int(tri_local[orig])
            if s[0].meshes[mi].is_area_light:
                var al_idx = Int(mesh_al_idx[mi])
                prim_ids_gpu[k].type          = Int8(3)
                prim_ids_gpu[k].id1           = Int64(al_idx)
                prim_ids_gpu[k].id2           = (Int64(mi) << 32) | Int64(ti)
                prim_ids_gpu[k].materialIndex = Int64(al_mat_base + al_idx)
            else:
                var mat_idx = Int(s[0].meshes[mi].mat_idx)
                prim_ids_gpu[k].type          = Int8(0)
                prim_ids_gpu[k].id1           = Int64(mi)
                prim_ids_gpu[k].id2           = Int64(ti * 3)
                prim_ids_gpu[k].materialIndex = Int64(mat_idx) if mat_idx >= 0 else Int64(default_mat_idx)
        else:
            var gidx = orig - Int(total_tris)
            var ci = Int(group_curve_idx[gidx])
            var curve_mat_idx = Int(s[0].curves_mat[ci])
            prim_ids_gpu[k].type          = Int8(5)
            prim_ids_gpu[k].id1           = Int64(ci)
            prim_ids_gpu[k].id2           = Int64(group_id2[gidx])
            prim_ids_gpu[k].materialIndex = Int64(curve_al_mat_idx[ci]) if s[0].curves_al[ci] else (Int64(curve_mat_idx) if curve_mat_idx >= 0 else Int64(default_mat_idx))
        prim_ids_gpu[k].instanceIdx = Int32(-1)
        prim_ids_gpu[k]._pad0 = Int8(0); prim_ids_gpu[k]._pad1 = Int8(0); prim_ids_gpu[k]._pad2 = Int8(0)

    for k in range(Int(total_prims)):
        var orig = Int(bvh_order[k])
        if orig < Int(total_tris):
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
                prim_ids[k].materialIndex = Int64(mat_idx) if mat_idx >= 0 else Int64(default_mat_idx)
        elif orig < Int(total_tris) + Int(total_curve_groups):
            var gidx = orig - Int(total_tris)
            var ci = Int(group_curve_idx[gidx])
            var curve_mat_idx = Int(s[0].curves_mat[ci])
            prim_ids[k].type          = Int8(5)
            prim_ids[k].id1           = Int64(ci)
            prim_ids[k].id2           = Int64(group_id2[gidx])
            prim_ids[k].materialIndex = Int64(curve_al_mat_idx[ci]) if s[0].curves_al[ci] else (Int64(curve_mat_idx) if curve_mat_idx >= 0 else Int64(default_mat_idx))
        else:
            var inst_idx = orig - Int(total_tris) - Int(total_curve_groups)
            prim_ids[k].type          = Int8(6)
            prim_ids[k].id1           = Int64(inst_idx)
            prim_ids[k].id2           = Int64(0)
            prim_ids[k].materialIndex = Int64(0)
        # instanceIdx is only meaningful on a *resolved* triangle hit (set by
        # traverse_bvh2_core's type==6 branch when it recurses into a BLAS) —
        # every ordinary top-level entry here, instance leaves included,
        # starts at -1.
        prim_ids[k].instanceIdx = Int32(-1)
        prim_ids[k]._pad0 = Int8(0); prim_ids[k]._pad1 = Int8(0)
        prim_ids[k]._pad2 = Int8(0)

    tri_mesh.free(); tri_local.free(); bvh_order.free(); bvh_order_gpu.free(); mesh_al_idx.free()
    curve_al_mat_idx.free()
    curve_group_base.free(); group_curve_idx.free(); group_id2.free()

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
    var tex_ptrs = alloc[UnsafePointer[UInt8, MutExternalOrigin]](max(n_tex, 1))
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
    psc[0].bvh_nodes        = bvh_nodes_gpu
    psc[0].prim_ids         = prim_ids_gpu
    psc[0].bvh_node_count   = node_count_gpu
    psc[0].prim_count       = total_prims_gpu
    psc[0].bvh_nodes_cpu      = bvh_nodes
    psc[0].prim_ids_cpu       = prim_ids
    psc[0].bvh_node_count_cpu = node_count
    psc[0].prim_count_cpu     = total_prims
    psc[0].blas_nodes_arr   = blas_nodes_arr
    psc[0].blas_primids_arr = blas_primids_arr
    psc[0].blas_node_counts   = blas_node_counts
    psc[0].blas_primid_counts = blas_primid_counts
    psc[0].blas_count       = Int32(n_templates)
    psc[0].instances        = instances_c
    psc[0].instance_count   = total_instances
    psc[0].template_mesh_start = template_mesh_start
    psc[0].template_mesh_end   = template_mesh_end
    psc[0].film_w           = s[0].film_w
    psc[0].film_h           = s[0].film_h
    psc[0].crop_x0          = s[0].crop_x0
    psc[0].crop_x1          = s[0].crop_x1
    psc[0].crop_y0          = s[0].crop_y0
    psc[0].crop_y1          = s[0].crop_y1
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
    psc[0].sppm_radius           = s[0].sppm_radius
    psc[0].sppm_photons_per_iter = s[0].sppm_photons_per_iter
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
        psc[0].distant_lights = UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling()
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
        psc[0].point_lights = UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling()
    psc[0].point_count = Int32(np2)

    var ni = len(s[0].inf_tex_idx)
    if ni > 0:
        var il_buf = alloc[InfiniteLight_C](ni)
        for i in range(ni):
            var tidx = s[0].inf_tex_idx[i]
            var sc = RGB(s[0].inf_rgb[i*3+0], s[0].inf_rgb[i*3+1], s[0].inf_rgb[i*3+2])
            var cdf_w = Int32(0); var cdf_h = Int32(0)
            var cdf_ptr = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
            var raw_pixels = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
            if tidx >= Int32(0):
                var fname2 = psc[0].tex_filenames[Int(tidx)]
                var pixels_ptr = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
                var iw_out = alloc[Int32](1); var ih_out = alloc[Int32](1)
                iw_out[0] = Int32(0); ih_out[0] = Int32(0)
                var load_ok = external_call["load_texture_rgb", Int32,
                    UnsafePointer[UInt8, MutExternalOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin],
                    UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
                    Int32](
                    fname2, pixels_ptr, iw_out, ih_out, Int32(0))
                var iw = Int(iw_out[0]); var ih = Int(ih_out[0])
                iw_out.free(); ih_out.free()
                if load_ok != Int32(0) and iw > 0 and ih > 0:
                    var pixels = pixels_ptr[0]
                    # Do NOT vertically flip here (removed a flip added in
                    # 73f368fd "fix upside-down environment"): in this equal-area
                    # octahedral mapping, row-flip (v -> 1-v) mirrors the decoded
                    # direction's Y axis, not Z/elevation, so it wasn't a valid
                    # "upside-down" fix. Confirmed wrong via cast-shadow direction
                    # (bunny-fur) vs env map luminance centroid; see
                    # project_infinite_light_shadows memory.
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
        psc[0].infinite_lights = UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling()
    psc[0].infinite_count = Int32(ni)

    # ---- Analytical spheres ----
    var ns = len(s[0].spheres_cx)
    if ns > 0:
        var sph_buf = alloc[Sphere_C](ns)
        for i in range(ns):
            var em = SampledSpectrum(s[0].spheres_rgb[i].r, s[0].spheres_rgb[i].g, s[0].spheres_rgb[i].b)
            var al_flag = Int8(1) if s[0].spheres_al[i] else Int8(0)
            var sph_mat_idx = s[0].spheres_mat[i]
            if sph_mat_idx == Int32(-1) and not s[0].spheres_al[i]:
                sph_mat_idx = default_mat_idx
            sph_buf[i] = Sphere_C(
                Point3f(s[0].spheres_cx[i], s[0].spheres_cy[i], s[0].spheres_cz[i]),
                s[0].spheres_r[i],
                sph_mat_idx,
                al_flag,
                Int8(0), Int8(0), Int8(0),
                em)
        psc[0].spheres = sph_buf
    else:
        psc[0].spheres = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling()
    psc[0].sphere_count = Int32(ns)

    # ---- Native curves (hair/fur) ----
    # n_pieces per curve (flatness-adaptive) was already computed above for
    # the per-piece BVH leaf construction — reuse it here instead of
    # recomputing.
    var nc = len(s[0].curves_mat)
    if nc > 0:
        var curve_buf = alloc[Curve_C](nc)
        for i in range(nc):
            var cb = i * 12
            curve_buf[i] = Curve_C(
                Point3f(s[0].curves_cp[cb+0], s[0].curves_cp[cb+1], s[0].curves_cp[cb+2]),
                Point3f(s[0].curves_cp[cb+3], s[0].curves_cp[cb+4], s[0].curves_cp[cb+5]),
                Point3f(s[0].curves_cp[cb+6], s[0].curves_cp[cb+7], s[0].curves_cp[cb+8]),
                Point3f(s[0].curves_cp[cb+9], s[0].curves_cp[cb+10], s[0].curves_cp[cb+11]),
                s[0].curves_w0[i], s[0].curves_w1[i], s[0].curves_mat[i], curve_n_pieces[i])
        psc[0].curves = curve_buf
    else:
        psc[0].curves = UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling()
    psc[0].curve_count = Int32(nc)
    curve_n_pieces.free()

    # ---- Curve area light NEE entries (al_list[n_al_mesh:]) ----
    # Mirrors the mesh area-light loop above, but needs curve_buf/n_pieces
    # (curve_light_tube_area), which only exist from this point on.
    var cl_running = Int32(0)
    for i in range(nc):
        if s[0].curves_al[i]:
            var idx = n_al_mesh + Int(cl_running)
            al_list[idx].meshIdx    = Int32(i)
            al_list[idx].n_tris     = Int32(0)
            al_list[idx].emission   = s[0].curves_al_rgb[i]
            al_list[idx].total_area = curve_light_tube_area(psc[0].curves[i])
            al_list[idx].kind       = Int8(1)
            cl_running += 1
    psc[0].area_light_count = Int32(n_al_mesh) + cl_running

    # ---- Heterogeneous density grids ("uniformgrid" media) ----
    var ng = len(s[0].grid_nx)
    if ng > 0:
        var grid_buf = alloc[Grid_C](ng)
        for i in range(ng):
            var nx = s[0].grid_nx[i]; var ny = s[0].grid_ny[i]; var nz = s[0].grid_nz[i]
            var n_voxels = Int(nx) * Int(ny) * Int(nz)
            var density_buf = alloc[Float32](max(n_voxels, 1))
            var base = Int(s[0].grid_density_base[i])
            var max_d = Float32(0.0)
            for vi in range(n_voxels):
                var dv = s[0].grid_density[base + vi]
                density_buf[vi] = dv
                if dv > max_d: max_d = dv
            var ctm_tmp = alloc[Float32](16)
            var w2m = alloc[Float32](16)
            for ci in range(16):
                ctm_tmp[ci] = s[0].grid_ctm[i*16 + ci]
            _ = matrix_invert(ctm_tmp, w2m)
            var w2m_simd = SIMD[DType.float32, 16](
                w2m[0], w2m[1], w2m[2], w2m[3], w2m[4], w2m[5], w2m[6], w2m[7],
                w2m[8], w2m[9], w2m[10], w2m[11], w2m[12], w2m[13], w2m[14], w2m[15])
            grid_buf[i] = Grid_C(
                density_buf, nx, ny, nz,
                Point3f(s[0].grid_p0[i*3], s[0].grid_p0[i*3+1], s[0].grid_p0[i*3+2]),
                Point3f(s[0].grid_p1[i*3], s[0].grid_p1[i*3+1], s[0].grid_p1[i*3+2]),
                w2m_simd, max_d)
            ctm_tmp.free(); w2m.free()
        psc[0].grids = grid_buf
    else:
        psc[0].grids = UnsafePointer[Grid_C, MutExternalOrigin].unsafe_dangling()
    psc[0].grid_count = Int32(ng)

    # ---- Media ----
    var nm = len(s[0].med_g)
    if nm > 0:
        var med_buf = alloc[Medium_C](nm)
        for i in range(nm):
            var sa = SampledSpectrum(s[0].med_sa[i*3], s[0].med_sa[i*3+1], s[0].med_sa[i*3+2])
            var ss = SampledSpectrum(s[0].med_ss[i*3], s[0].med_ss[i*3+1], s[0].med_ss[i*3+2])
            med_buf[i] = Medium_C(sa, ss, s[0].med_g[i],
                                  s[0].med_grid_idx[i], Float32(0), Float32(0))
        psc[0].mediums = med_buf
    else:
        psc[0].mediums = UnsafePointer[Medium_C, MutExternalOrigin].unsafe_dangling()
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

def resize_film(psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
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

def mojo_parse_scene(path: UnsafePointer[UInt8, MutExternalOrigin],
                     verbose: Bool = False,
                    ) -> UnsafePointer[ParsedScene_Mojo, MutExternalOrigin]:
    external_call["createTextureSystem", NoneType]()
    var handle = scanner_open(path)
    if handle[0].is_at_end != Int32(0):
        print("Error: cannot open scene file:", String(unsafe_from_utf8_ptr=path.as_immutable()))
        scanner_free(handle)
        return UnsafePointer[ParsedScene_Mojo, MutExternalOrigin].unsafe_dangling()

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

    var psc = alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, verbose)
    _ = s_ptr.take_pointee()
    s_ptr.free()
    return psc

def mojo_parsed_free(psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin]):
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
    if psc[0].bvh_node_count_cpu > 0:
        psc[0].bvh_nodes_cpu.free()
    if psc[0].prim_count_cpu > 0:
        psc[0].prim_ids_cpu.free()
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
            if _is_real_ptr(il.cdf_ptr):
                il.cdf_ptr.free()
            if _is_real_ptr(il.pixels_ptr):
                _ = external_call["free_texture_rgb", Int32,
                    UnsafePointer[Float32, MutExternalOrigin]](il.pixels_ptr)
            il.world_to_light.free()
        psc[0].infinite_lights.free()
    if psc[0].sphere_count > 0:
        psc[0].spheres.free()
    if psc[0].curve_count > 0:
        psc[0].curves.free()
    if psc[0].measured_count > 0:
        # Per-pointer sentinel-address guards (matches light_sampler.cdf's
        # convention above): a MeasuredBRDF_C for a file that FAILED to load
        # has every pointer field set via unsafe_dangling() (see
        # measured_bsdf.mojo's _fail()), which must never be passed to
        # .free() directly.
        for mi in range(Int(psc[0].measured_count)):
            var mb = psc[0].measured_brdfs[mi]
            if Int(mb.theta_i) > 4: mb.theta_i.free()
            if Int(mb.phi_i) > 4: mb.phi_i.free()
            if Int(mb.wavelengths) > 4: mb.wavelengths.free()
            if Int(mb.ndf_data) > 4: mb.ndf_data.free()
            if Int(mb.sigma_data) > 4: mb.sigma_data.free()
            if Int(mb.vndf_data) > 4: mb.vndf_data.free()
            if Int(mb.vndf_marg) > 4: mb.vndf_marg.free()
            if Int(mb.vndf_cond) > 4: mb.vndf_cond.free()
            if Int(mb.lum_data) > 4: mb.lum_data.free()
            if Int(mb.lum_marg) > 4: mb.lum_marg.free()
            if Int(mb.lum_cond) > 4: mb.lum_cond.free()
            if Int(mb.spectra_data) > 4: mb.spectra_data.free()
        psc[0].measured_brdfs.free()
    if psc[0].grid_count > 0:
        for gi in range(Int(psc[0].grid_count)):
            psc[0].grids[gi].density.free()
        psc[0].grids.free()
    if Int(psc[0].light_sampler.cdf) > 4:
        psc[0].light_sampler.cdf.free()
    # blas_nodes_arr/blas_primids_arr/instances are always real allocations
    # (min size 1, see finalize_scene) regardless of blas_count/instance_count.
    for bi in range(Int(psc[0].blas_count)):
        psc[0].blas_nodes_arr[bi].free()
        psc[0].blas_primids_arr[bi].free()
    psc[0].blas_nodes_arr.free()
    psc[0].blas_primids_arr.free()
    psc[0].blas_node_counts.free()
    psc[0].blas_primid_counts.free()
    psc[0].instances.free()
    psc[0].template_mesh_start.free()
    psc[0].template_mesh_end.free()
    psc.free()

def mojo_apply_overrides(
    psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
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
    psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
    spectral: SpectralHandle,
) -> UnsafePointer[SceneDescriptor2_C, MutExternalOrigin]:
    var sd = alloc[SceneDescriptor2_C](1)
    sd[0].bvh2Nodes        = psc[0].bvh_nodes_cpu
    sd[0].primIds          = psc[0].prim_ids_cpu
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
    sd[0].curves           = psc[0].curves
    sd[0].curveCount       = Int64(psc[0].curve_count)
    sd[0].mediums          = psc[0].mediums
    sd[0].mediumCount      = Int64(psc[0].medium_count)
    sd[0].mediumInterfaces = psc[0].medium_ifaces
    sd[0].mediumIfaceCount = Int64(psc[0].medium_iface_count)
    sd[0].grids            = psc[0].grids
    sd[0].gridCount        = Int64(psc[0].grid_count)
    sd[0].lightSampler    = psc[0].light_sampler
    sd[0].blasNodesArr    = psc[0].blas_nodes_arr
    sd[0].blasPrimIdsArr  = psc[0].blas_primids_arr
    sd[0].blasCount       = Int64(psc[0].blas_count)
    sd[0].instances       = psc[0].instances
    sd[0].instanceCount   = Int64(psc[0].instance_count)
    sd[0].measuredBrdfs      = psc[0].measured_brdfs
    sd[0].measuredBrdfCount  = Int64(psc[0].measured_count)
    sd[0].spectral        = spectral
    # CPU path never needs the GPU-resident texture array (shading.mojo's
    # _tex_lookup[False] branch uses sd.textures/textureCount above
    # instead) -- dangling/0, same convention every other GPU-only field
    # here would use if this were a GPU builder.
    sd[0].gpuTextures      = UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling()
    sd[0].gpuTextureCount  = Int64(0)
    return sd
