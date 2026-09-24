from std.collections import Array
from std.ffi import external_call
from std.time import perf_counter_ns
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memcpy
from std.math import tan, sqrt, abs
from std.atomic import Atomic
from std.sys.info import num_performance_cores
from max.algorithm import parallelize
from std.subprocess import run
from std.os.path import exists
from .diagnostics import warn_unsupported, warn_unsupported_in
from .lexer import (PbrtScanner, scanner_open, scanner_free, scanner_is_at_end,
                    scanner_scan_token, scanner_parse_quoted_string,
                    scanner_scan_char, scanner_scan_float,
                    _psc_streq,
                    _psc_scan_spectrum_scalar, _psc_collect_params, ParameterDictionary,
                    _psc_skip_params, _psc_skip_line)
from .parse_types import (SceneParseState, MeshAccum, NamedMaterial, scene_path,
                           ctm_push, ctm_pop, PSC_NAME_MAX, PSC_FILE_MAX)
from .geometry import (RGB, Point3f, Vec3f, Material_C, MatKind, AreaLight_C,
                        Sphere_C, Curve_C, CURVE_N_PIECES, curve_piece_bounds, curve_bspline_point, curve_light_tube_area, dot, DistantLight_C, PointLight_C, InfiniteLight_C,
                        TriangleMesh_C, PrimId_C, Medium_C, MediumInterface_C, Grid_C, NvdbGrid_C, PI,
                        LightSampler_C, Instance_C, MeasuredBRDF_C, GpuTexture_C, NormalSlopeMap_C, normal_slope_map_none, _is_real_ptr)
from .nanovdb import nvdb_load, nvdb_load_named, nvdb_data, nvdb_size, nvdb_free, nvdb_index_bbox, nvdb_value_range, nvdb_map_invmatf, nvdb_map_vecf
from .noise import _perlin_perm_table, cloud_density
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
    var raster_to_camera: Pointer[Float32, MutUntrackedOrigin]   # 16 floats, column-major
    var camera_to_world:  Pointer[Float32, MutUntrackedOrigin]   # 16 floats, column-major
    var materials:        Pointer[Material_C, MutUntrackedOrigin]
    var material_count:   Int32
    var area_lights:      Pointer[AreaLight_C, MutUntrackedOrigin]
    var area_light_count: Int32
    var meshes:           Pointer[TriangleMesh_C, MutUntrackedOrigin]
    var mesh_pts:         Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var mesh_vis:         Pointer[Pointer[Int64, MutUntrackedOrigin], MutUntrackedOrigin]
    var mesh_fis:         Pointer[Pointer[Int64, MutUntrackedOrigin], MutUntrackedOrigin]
    var mesh_n_verts:     Pointer[Int32, MutUntrackedOrigin]
    var mesh_n_tris:      Pointer[Int32, MutUntrackedOrigin]
    var mesh_uv_n_verts:  Pointer[Int32, MutUntrackedOrigin]  # per-mesh UV vertex count; 0 = no UVs
    var mesh_nrm_n_verts: Pointer[Int32, MutUntrackedOrigin]  # per-mesh normal vertex count; 0 = no shading normals
    var mesh_count:       Int32
    var bvh_nodes:        Pointer[BVH2Node, MutUntrackedOrigin]   # GPU-safe TLAS: tris+curves only, no instance leaves
    var prim_ids:         Pointer[PrimId_C, MutUntrackedOrigin]
    var bvh_node_count:   Int32
    var prim_count:       Int32
    # CPU-inclusive TLAS: tris+curves+instances. Used only by
    # mojo_parsed_scene_descriptor (SceneDescriptor2_C, the CPU render path).
    # GPU's device-side upload always reads bvh_nodes/prim_ids above instead —
    # its traversal kernels have no BLAS/instance buffers to resolve a
    # PrimId_C.type==6 leaf, so one must never appear in its uploaded arrays
    # (confirmed via testing: it does not degrade gracefully, it crashes).
    var bvh_nodes_cpu:      Pointer[BVH2Node, MutUntrackedOrigin]
    var prim_ids_cpu:       Pointer[PrimId_C, MutUntrackedOrigin]
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
    var film_exposuretime: Float32
    var film_wb:          SIMD[DType.float32, 16]  # 3x3 row-major in lanes 0..8; sensor white balance
    var film_max_comp:    Float32
    var film_filename:    Pointer[UInt8, MutUntrackedOrigin]      # null-terminated
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
    var tex_filenames:    Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin]
    var tex_count:        Int32
    # Parallel to `tex_filenames`: the slope-space form of every texture some
    # material uses as a NORMAL map, for the SMS/MNEE manifold walk (see
    # geometry.mojo's NormalSlopeMap_C). Entries for other textures have
    # res == 0.
    var nmaps:            Pointer[NormalSlopeMap_C, MutUntrackedOrigin]
    var distant_lights:   Pointer[DistantLight_C, MutUntrackedOrigin]
    var distant_count:    Int32
    var point_lights:     Pointer[PointLight_C, MutUntrackedOrigin]
    var point_count:      Int32
    var infinite_lights:  Pointer[InfiniteLight_C, MutUntrackedOrigin]
    var infinite_count:   Int32
    var spheres:          Pointer[Sphere_C, MutUntrackedOrigin]
    var sphere_count:     Int32
    var curves:           Pointer[Curve_C, MutUntrackedOrigin]
    var curve_count:      Int32
    var mediums:          Pointer[Medium_C, MutUntrackedOrigin]
    var medium_count:     Int32
    var medium_ifaces:    Pointer[MediumInterface_C, MutUntrackedOrigin]
    var medium_iface_count: Int32
    var grids:            Pointer[Grid_C, MutUntrackedOrigin]
    var grid_count:       Int32
    var nvdb_grids:       Pointer[NvdbGrid_C, MutUntrackedOrigin]
    var nvdb_grid_count:  Int32
    var light_sampler:    LightSampler_C
    # Object instancing: one BLAS (private BVH2, over `meshes` above) per
    # ObjectBegin/ObjectEnd template, referenced by Instance_C.blasIdx.
    var blas_nodes_arr:   Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin]
    var blas_primids_arr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin]
    var blas_node_counts:   Pointer[Int32, MutUntrackedOrigin]  # per-BLAS array length, needed for GPU upload
    var blas_primid_counts: Pointer[Int32, MutUntrackedOrigin]
    var blas_count:       Int32
    var instances:        Pointer[Instance_C, MutUntrackedOrigin]
    var instance_count:   Int32
    # Mesh-index range [start, end) each template's BLAS spans, into the SAME
    # `meshes` array above (a template can bundle several Shape calls, e.g.
    # barcelona-pavilion's tree templates each have 5-9 plymesh shapes).
    # Only consumer today: the Vulkan RT interop scene builder (pipeline.mojo),
    # which needs to know which mesh indices are template-only (excluded from
    # its ordinary one-BLAS-per-mesh loop) and which meshes feed which
    # per-template multi-geometry BLAS -- see [[project_vulkan_rt_backend]].
    var template_mesh_start: Pointer[Int32, MutUntrackedOrigin]
    var template_mesh_end:   Pointer[Int32, MutUntrackedOrigin]
    # "measured" materials: one MeasuredBRDF_C per distinct .bsdf file
    # (deduped by path), referenced by Material_C.measured_idx. Populated at
    # final-scene-build time from named_materials[i].measured_bsdf_path -- see
    # the dedup+load loop near the materials array build below.
    var measured_brdfs:  Pointer[MeasuredBRDF_C, MutUntrackedOrigin]
    var measured_count:  Int32

# ── Matrix utilities ──────────────────────────────────────────────────────────

def _psc_identity(m: Pointer[Float32, MutUntrackedOrigin]):
    for i in range(16):
        m[unsafe_offset=i] = Float32(0)
    m[unsafe_offset=0] = Float32(1)
    m[unsafe_offset=5] = Float32(1)
    m[unsafe_offset=10] = Float32(1)
    m[unsafe_offset=15] = Float32(1)

def _psc_matcopy(dst: Pointer[Float32, MutUntrackedOrigin],
                src: Pointer[Float32, MutUntrackedOrigin]):
    for i in range(16):
        dst[unsafe_offset=i] = src[unsafe_offset=i]

def _psc_ctm_concat(s: Pointer[SceneParseState, MutUntrackedOrigin],
                   t: Pointer[Float32, MutUntrackedOrigin]):
    """Compute s.ctm = s.ctm × t and store back."""
    var result = unsafe_alloc[Float32](16)
    matrix_multiply(s[unsafe_offset=0].ctm.unsafe_ptr(), t, result)
    for i in range(16):
        s[unsafe_offset=0].ctm[i] = result[unsafe_offset=i]
    result.unsafe_free()

def _psc_row_to_col(col_out: Pointer[Float32, MutUntrackedOrigin],
                   row_in:  Pointer[Float32, MutUntrackedOrigin]):
    for row in range(4):
        for col in range(4):
            col_out[unsafe_offset=col * 4 + row] = row_in[unsafe_offset=row * 4 + col]

# ── Transform keyword handlers ────────────────────────────────────────────────

def _psc_handle_translate(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                         s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """Translate tx ty tz  →  CTM = CTM × T(tx,ty,tz)"""
    var v = unsafe_alloc[Float32](3)
    v[unsafe_offset=0] = Float32(0); v[unsafe_offset=1] = Float32(0); v[unsafe_offset=2] = Float32(0)
    _ = scanner_scan_float(handle, v.unsafe_offset(0))
    _ = scanner_scan_float(handle, v.unsafe_offset(1))
    _ = scanner_scan_float(handle, v.unsafe_offset(2))
    var t = unsafe_alloc[Float32](16)
    _psc_identity(t)
    t[unsafe_offset=12] = v[unsafe_offset=0]; t[unsafe_offset=13] = v[unsafe_offset=1]; t[unsafe_offset=14] = v[unsafe_offset=2]   # col-major: col3 = (tx,ty,tz,1)
    _psc_ctm_concat(s, t)
    v.unsafe_free(); t.unsafe_free()

def _psc_handle_scale_kw(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                        s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """Scale sx sy sz  →  CTM = CTM × S(sx,sy,sz)"""
    var v = unsafe_alloc[Float32](3)
    v[unsafe_offset=0] = Float32(1); v[unsafe_offset=1] = Float32(1); v[unsafe_offset=2] = Float32(1)
    _ = scanner_scan_float(handle, v.unsafe_offset(0))
    _ = scanner_scan_float(handle, v.unsafe_offset(1))
    _ = scanner_scan_float(handle, v.unsafe_offset(2))
    var t = unsafe_alloc[Float32](16)
    _psc_identity(t)
    t[unsafe_offset=0] = v[unsafe_offset=0]; t[unsafe_offset=5] = v[unsafe_offset=1]; t[unsafe_offset=10] = v[unsafe_offset=2]     # col-major: diagonal
    _psc_ctm_concat(s, t)
    v.unsafe_free(); t.unsafe_free()

def _psc_handle_rotate(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                      s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """Rotate angle ax ay az  →  CTM = CTM × R(angle, axis)"""
    from std.math import sin as _sin, cos as _cos, sqrt as _sqrt
    var rv = unsafe_alloc[Float32](4)  # angle, ax, ay, az
    rv[unsafe_offset=0] = Float32(0); rv[unsafe_offset=1] = Float32(0); rv[unsafe_offset=2] = Float32(0); rv[unsafe_offset=3] = Float32(1)
    _ = scanner_scan_float(handle, rv.unsafe_offset(0))
    _ = scanner_scan_float(handle, rv.unsafe_offset(1))
    _ = scanner_scan_float(handle, rv.unsafe_offset(2))
    _ = scanner_scan_float(handle, rv.unsafe_offset(3))
    var angle = rv[unsafe_offset=0] * PI / Float32(180)
    var ax = rv[unsafe_offset=1]; var ay = rv[unsafe_offset=2]; var az = rv[unsafe_offset=3]
    var ln = _sqrt(ax*ax + ay*ay + az*az)
    if ln > Float32(1e-12): ax /= ln; ay /= ln; az /= ln
    var c = _cos(angle); var sv = _sin(angle); var mc = Float32(1) - c
    var t = unsafe_alloc[Float32](16)
    # Column-major rotation matrix (standard Rodrigues)
    t[unsafe_offset=0]  = c + ax*ax*mc;       t[unsafe_offset=1]  = ay*ax*mc + az*sv;   t[unsafe_offset=2]  = az*ax*mc - ay*sv;   t[unsafe_offset=3]  = Float32(0)
    t[unsafe_offset=4]  = ax*ay*mc - az*sv;   t[unsafe_offset=5]  = c + ay*ay*mc;       t[unsafe_offset=6]  = az*ay*mc + ax*sv;   t[unsafe_offset=7]  = Float32(0)
    t[unsafe_offset=8]  = ax*az*mc + ay*sv;   t[unsafe_offset=9]  = ay*az*mc - ax*sv;   t[unsafe_offset=10] = c + az*az*mc;        t[unsafe_offset=11] = Float32(0)
    t[unsafe_offset=12] = Float32(0);         t[unsafe_offset=13] = Float32(0);          t[unsafe_offset=14] = Float32(0);          t[unsafe_offset=15] = Float32(1)
    _psc_ctm_concat(s, t)
    rv.unsafe_free(); t.unsafe_free()

def _psc_handle_lookat(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                      s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """LookAt ex ey ez  lx ly lz  ux uy uz"""
    from std.math import sqrt as _sqrt
    var v = unsafe_alloc[Float32](9)
    for i in range(9): _ = scanner_scan_float(handle, v.unsafe_offset(i))
    var ex = v[unsafe_offset=0]; var ey = v[unsafe_offset=1]; var ez = v[unsafe_offset=2]
    var lx = v[unsafe_offset=3]; var ly = v[unsafe_offset=4]; var lz = v[unsafe_offset=5]
    var ux = v[unsafe_offset=6]; var uy = v[unsafe_offset=7]; var uz = v[unsafe_offset=8]
    v.unsafe_free()

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

    var t = unsafe_alloc[Float32](16)
    t[unsafe_offset=0]  = rx;  t[unsafe_offset=1]  = nx;  t[unsafe_offset=2]  = dx;  t[unsafe_offset=3]  = Float32(0)
    t[unsafe_offset=4]  = ry;  t[unsafe_offset=5]  = ny;  t[unsafe_offset=6]  = dy;  t[unsafe_offset=7]  = Float32(0)
    t[unsafe_offset=8]  = rz;  t[unsafe_offset=9]  = nz;  t[unsafe_offset=10] = dz;  t[unsafe_offset=11] = Float32(0)
    t[unsafe_offset=12] = -(rx*ex + ry*ey + rz*ez)
    t[unsafe_offset=13] = -(nx*ex + ny*ey + nz*ez)
    t[unsafe_offset=14] = -(dx*ex + dy*ey + dz*ez)
    t[unsafe_offset=15] = Float32(1)
    _psc_ctm_concat(s, t)
    t.unsafe_free()

# ── Directive handlers ────────────────────────────────────────────────────────

def _psc_handle_integrator(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                          s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.unsafe_free()
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].max_depth = params.get_int("maxdepth", s[unsafe_offset=0].max_depth)
    s[unsafe_offset=0].sppm_radius = params.get_float("radius", s[unsafe_offset=0].sppm_radius)
    s[unsafe_offset=0].sppm_photons_per_iter = params.get_int("photonsperiteration", s[unsafe_offset=0].sppm_photons_per_iter)

def _psc_handle_sampler(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                       s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.unsafe_free()
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].samples_per_pixel = params.get_int("pixelsamples", s[unsafe_offset=0].samples_per_pixel)
    s[unsafe_offset=0].samples_per_pixel = params.get_int("samples", s[unsafe_offset=0].samples_per_pixel)

def _psc_handle_filter(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                      s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    # Default radius PER TYPE, as pbrt-v4 has it (filters.cpp): box 0.5,
    # triangle 2, gaussian 1.5. Every type used to inherit the Gaussian's 1.5,
    # so a bare `PixelFilter "box"` blurred over 3 pixels instead of 1.
    var default_radius = Float32(1.5)
    if _psc_streq(sbuf, "triangle") or _psc_streq(sbuf, "tent"):
        s[unsafe_offset=0].filter_type = Int32(1)
        default_radius = Float32(2.0)
    elif _psc_streq(sbuf, "box"):
        s[unsafe_offset=0].filter_type = Int32(2)
        default_radius = Float32(0.5)
    else:
        # Anything else renders as a Gaussian. Say so, rather than let a
        # scene asking for mitchell/lanczos/sinc believe it got one -- nothing
        # in the corpus does today, which is exactly when a silent fallback
        # goes unnoticed.
        if not _psc_streq(sbuf, "gaussian"):
            print("Warning: PixelFilter type not implemented -- rendering with"
                  + " a Gaussian filter instead. Supported: gaussian, box,"
                  + " triangle.")
        s[unsafe_offset=0].filter_type = Int32(0)
    sbuf.unsafe_free()
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].filter_support_x = params.get_float("xradius", default_radius)
    s[unsafe_offset=0].filter_support_y = params.get_float("yradius", default_radius)
    s[unsafe_offset=0].filter_sigma = params.get_float("sigma", s[unsafe_offset=0].filter_sigma)

def _psc_handle_film(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                    s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    # The film TYPE is otherwise unused: every type renders as "rgb". That is
    # right for "rgb"/"spectral", but pbrt's "gbuffer" film also writes
    # auxiliary AOV channels (albedo/normal/depth/variance) into the same EXR,
    # which gonzales does not. Say so rather than let a scene author believe
    # those channels were produced -- watercolor and kroken both ask for it.
    var is_gbuf = _psc_streq(sbuf, "gbuffer")
    sbuf.unsafe_free()
    if is_gbuf:
        print("Warning: Film \"gbuffer\" — rendering as \"rgb\"; the auxiliary"
              + " G-buffer channels (albedo/normal/depth/variance) are NOT"
              + " written into the output EXR.")
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].film_w = params.get_int("xresolution", s[unsafe_offset=0].film_w)
    s[unsafe_offset=0].film_h = params.get_int("yresolution", s[unsafe_offset=0].film_h)
    s[unsafe_offset=0].film_filename = params.get_string("filename", s[unsafe_offset=0].film_filename)
    s[unsafe_offset=0].film_iso = params.get_float("iso", s[unsafe_offset=0].film_iso)
    s[unsafe_offset=0].film_exposuretime = params.get_float("exposuretime", s[unsafe_offset=0].film_exposuretime)
    s[unsafe_offset=0].film_whitebalance = params.get_float("whitebalance", s[unsafe_offset=0].film_whitebalance)
    var sensor_str = params.get_string("sensor", "cie1931")
    if sensor_str != "cie1931" and sensor_str != "":
        # pbrt models a named sensor with its MEASURED per-wavelength r/g/b
        # response curves, fitted to XYZ through 24 Macbeth swatch spectra.
        # Those tables are not ported, so fall back to pbrt's own default
        # sensor ("cie1931", i.e. the XYZ matching functions). Exposure and
        # white balance below are still applied exactly, so this differs from
        # pbrt only by the sensor's colour-response shape.
        print("Warning: film sensor '" + sensor_str + "' not modelled — using cie1931 response (exposure and whitebalance still applied).")
    s[unsafe_offset=0].film_sensor = sensor_str
    s[unsafe_offset=0].film_max_comp = params.get_float("maxcomponentvalue", s[unsafe_offset=0].film_max_comp)
    var cw = params.get_floats("cropwindow")
    if len(cw) >= 4:
        s[unsafe_offset=0].crop_x0 = cw[0]; s[unsafe_offset=0].crop_x1 = cw[1]
        s[unsafe_offset=0].crop_y0 = cw[2]; s[unsafe_offset=0].crop_y1 = cw[3]

def _psc_handle_camera(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                      s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var sbuf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    sbuf.unsafe_free()
    # Copy current CTM into cam2w_raw
    for i in range(16): s[unsafe_offset=0].cam2w_raw[i] = s[unsafe_offset=0].ctm[i]
    var params = _psc_collect_params(handle)
    s[unsafe_offset=0].camera_fov = params.get_float("fov", s[unsafe_offset=0].camera_fov)

def _psc_handle_transform(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                         s: Pointer[SceneParseState, MutUntrackedOrigin]):
    _ = scanner_scan_char(handle, UInt8(91))  # '['
    var tmp = unsafe_alloc[Float32](1)
    for i in range(16):
        _ = scanner_scan_float(handle, tmp)
        s[unsafe_offset=0].ctm[i] = tmp[unsafe_offset=0]
    tmp.unsafe_free()
    _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_handle_world_begin(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    for i in range(16): s[unsafe_offset=0].ctm[i] = Float32(0)
    s[unsafe_offset=0].ctm[0] = Float32(1); s[unsafe_offset=0].ctm[5] = Float32(1)
    s[unsafe_offset=0].ctm[10] = Float32(1); s[unsafe_offset=0].ctm[15] = Float32(1)
    s[unsafe_offset=0].ctm_stack.clear()

def _psc_handle_attribute_begin(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    ctm_push(s[unsafe_offset=0])
    s[unsafe_offset=0].attr_stack.append(s[unsafe_offset=0].cur_attr)

def _psc_handle_attribute_end(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    ctm_pop(s[unsafe_offset=0])
    if len(s[unsafe_offset=0].attr_stack) > 0:
        s[unsafe_offset=0].cur_attr = s[unsafe_offset=0].attr_stack[len(s[unsafe_offset=0].attr_stack) - 1]
        _ = s[unsafe_offset=0].attr_stack.pop()

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

def handle_curve_shape(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                            s: Pointer[SceneParseState, MutUntrackedOrigin]):
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
    var raw4 = unsafe_alloc[Float32](n_raw * 4)
    var xfm4 = unsafe_alloc[Float32](n_raw * 4)
    for i in range(n_raw):
        raw4[unsafe_offset=i*4+0] = cp_list[i*3+0]; raw4[unsafe_offset=i*4+1] = cp_list[i*3+1]
        raw4[unsafe_offset=i*4+2] = cp_list[i*3+2]; raw4[unsafe_offset=i*4+3] = Float32(1)
    transform_points(s[unsafe_offset=0].ctm.unsafe_ptr(), raw4, Int32(n_raw), xfm4)
    raw4.unsafe_free()

    # Split into (n_cp - 3) local B-spline segments: window i uses raw
    # control points [i, i+1, i+2, i+3] — matches the standard uniform
    # cubic B-spline curve-chain convention (same windowing PBRT itself uses).
    var n_seg = n_raw - 3
    var mat_idx = s[unsafe_offset=0].cur_attr.mat_idx
    for seg in range(n_seg):
        var t0 = Float32(seg) / Float32(n_seg)
        var t1 = Float32(seg + 1) / Float32(n_seg)
        for k in range(4):
            var vi = seg + k
            s[unsafe_offset=0].curves_cp.append(xfm4[unsafe_offset=vi*4+0])
            s[unsafe_offset=0].curves_cp.append(xfm4[unsafe_offset=vi*4+1])
            s[unsafe_offset=0].curves_cp.append(xfm4[unsafe_offset=vi*4+2])
        s[unsafe_offset=0].curves_w0.append(width0 + (width1 - width0) * t0)
        s[unsafe_offset=0].curves_w1.append(width0 + (width1 - width0) * t1)
        s[unsafe_offset=0].curves_mat.append(mat_idx)
        s[unsafe_offset=0].curves_al.append(s[unsafe_offset=0].cur_attr.is_alight)
        s[unsafe_offset=0].curves_al_rgb.append(s[unsafe_offset=0].cur_attr.al_rgb)
    xfm4.unsafe_free()

# ── Medium handlers ───────────────────────────────────────────────────────────

# Resolution the procedural "cloud" medium (see handle_named_medium's
# is_cloud branch) is baked to. pbrt evaluates its noise live, at whatever
# resolution the ray marcher happens to sample; baking commits to one fixed
# resolution up front.
#
# CORRECTED 2026-09-10: the original 160 was sized assuming noise frequency
# is relative to a 1-UNIT box (`frequency * 1.99^4 =~ 78 cycles across "the
# unit box"`). That was never the right frame -- pbrt's own Density(p) has
# no reference to p0/p1 at all; `frequency` is cycles per WORLD UNIT, full
# stop, and the box being baked into now correctly matches the medium's
# real geometric extent (see _resolve_pending_cloud_grid -- p0/p1 auto-size
# to the enclosing shape's AABB when the scene omits them, instead of
# silently defaulting to pbrt's own 1-unit-box default and clipping most of
# a several-unit-wide medium to zero density). `clouds.pbrt`'s bounding
# sphere has DIAMETER 2, so the finest octave actually spans
# `5*1.99^4*2 =~ 156` cycles across the true baked extent -- double the
# original estimate, and 160 samples/axis was barely above bare Nyquist
# for that, not "comfortably above" it: trilinear reconstruction needs real
# oversampling margin (bare Nyquist still looks heavily blurred after
# trilinear filtering, which is a poor lowpass compared to ideal sinc
# reconstruction). 320 is a ~2x oversample of the corrected 156-cycle
# figure (~4.6x samples/cycle) -- 320^3 = ~32.8M voxels, ~131MB, still
# comfortably parse-time-tractable. Not tied to any specific scene's
# `frequency`/extent -- a scene requesting a higher frequency or spanning a
# larger shape would need a higher CLOUD_BAKE_RES to stay crisp.
comptime CLOUD_BAKE_RES: Int = 320

def handle_named_medium(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                                  s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var name_buf = unsafe_alloc[UInt8](64)
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
    var is_nvdb = type_str == "nanovdb"
    var is_cloud = type_str == "cloud"

    # sigma_a/sigma_s: rgb triple, OR inline numeric spectrum array (mean of
    # samples, replicated to all 3 channels) via the same float-or-rgb
    # duality as texture "value"/"tex1"/"tex2" above -- EXCEPT a "spectrum"-
    # declared inline array (an even count of wavelength/value pairs, e.g.
    # "spectrum sigma_a" [200 .01 900 .01]) is NOT an RGB triple even when
    # it happens to have >=3 floats; _psc_get_sigma_or_rgb (not the shared
    # texture helper) handles that distinction -- see its own docstring for
    # the real bug this fixed (an RGB triple was being read off the raw
    # wavelength/value numbers). A named-spectrum string reference (0
    # floats collected) is silently unsupported, same as before -- no
    # medium in this scene corpus uses one.
    var sa_set = len(params.get_floats("sigma_a")) > 0
    var ss_set = len(params.get_floats("sigma_s")) > 0
    var sa = _psc_get_sigma_or_rgb(params, "sigma_a", RGB(Float32(0)))
    var ss = _psc_get_sigma_or_rgb(params, "sigma_s", RGB(Float32(0)))
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
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_imm())
        s[unsafe_offset=0].med_names.append(name_str)
        s[unsafe_offset=0].med_sa.append(sa.r * scale)
        s[unsafe_offset=0].med_sa.append(sa.g * scale)
        s[unsafe_offset=0].med_sa.append(sa.b * scale)
        s[unsafe_offset=0].med_ss.append(ss.r * scale)
        s[unsafe_offset=0].med_ss.append(ss.g * scale)
        s[unsafe_offset=0].med_ss.append(ss.b * scale)
        s[unsafe_offset=0].med_g.append(g_val)
        s[unsafe_offset=0].med_grid_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_temp_idx.append(Int32(-1))
        s[unsafe_offset=0].med_le_scale.append(Float32(0)); s[unsafe_offset=0].med_temp_offset.append(Float32(0)); s[unsafe_offset=0].med_temp_scale.append(Float32(1))
        s[unsafe_offset=0].med_is_sss.append(Int32(0))
    elif is_grid:
        # PBRT-v4 default for GridMedium sigma_a/sigma_s when unspecified is
        # ConstantSpectrum(1) (see media.cpp) — unlike gonzales's existing
        # HomogeneousMedium path above, which happens to default to 0 (a
        # pre-existing, separate behavior not touched here).
        var sa_eff = sa if sa_set else RGB(Float32(1))
        var ss_eff = ss if ss_set else RGB(Float32(1))
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_imm())
        s[unsafe_offset=0].med_names.append(name_str)
        s[unsafe_offset=0].med_sa.append(sa_eff.r * scale); s[unsafe_offset=0].med_sa.append(sa_eff.g * scale); s[unsafe_offset=0].med_sa.append(sa_eff.b * scale)
        s[unsafe_offset=0].med_ss.append(ss_eff.r * scale); s[unsafe_offset=0].med_ss.append(ss_eff.g * scale); s[unsafe_offset=0].med_ss.append(ss_eff.b * scale)
        s[unsafe_offset=0].med_g.append(g_val)
        s[unsafe_offset=0].med_grid_idx.append(Int32(len(s[unsafe_offset=0].grid_nx)))
        s[unsafe_offset=0].med_nvdb_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_temp_idx.append(Int32(-1))
        s[unsafe_offset=0].med_le_scale.append(Float32(0)); s[unsafe_offset=0].med_temp_offset.append(Float32(0)); s[unsafe_offset=0].med_temp_scale.append(Float32(1))
        s[unsafe_offset=0].med_is_sss.append(Int32(0))

        s[unsafe_offset=0].grid_nx.append(g_nx); s[unsafe_offset=0].grid_ny.append(g_ny); s[unsafe_offset=0].grid_nz.append(g_nz)
        s[unsafe_offset=0].grid_p0.append(g_p0.r); s[unsafe_offset=0].grid_p0.append(g_p0.g); s[unsafe_offset=0].grid_p0.append(g_p0.b)
        s[unsafe_offset=0].grid_p1.append(g_p1.r); s[unsafe_offset=0].grid_p1.append(g_p1.g); s[unsafe_offset=0].grid_p1.append(g_p1.b)
        for ci in range(16):
            s[unsafe_offset=0].grid_ctm.append(s[unsafe_offset=0].ctm[ci])
        s[unsafe_offset=0].grid_density_base.append(Int32(len(s[unsafe_offset=0].grid_density)))
        var expected_n = Int(g_nx) * Int(g_ny) * Int(g_nz)
        var copy_n = min(len(g_density), expected_n) if expected_n > 0 else len(g_density)
        for di in range(copy_n):
            s[unsafe_offset=0].grid_density.append(g_density[di])
        # Pad with zeros if the file had fewer values than nx*ny*nz declared
        # (shouldn't happen for well-formed scenes, but keeps indexing safe).
        for _ in range(copy_n, expected_n):
            s[unsafe_offset=0].grid_density.append(Float32(0))
    elif is_cloud:
        # PBRT-v4's procedural Perlin-noise CloudMedium. No baked asset file
        # (unlike "nanovdb") -- density is a live noise function of world
        # position, see noise.mojo's cloud_density (ported line-for-line
        # from real pbrt source, not reconstructed from memory). Baked HERE,
        # at parse time, into an ordinary dense grid -- i.e. this reuses the
        # exact "uniformgrid" branch just above verbatim (same grid_*
        # arrays, same med_grid_idx indexing), just with the density array
        # populated by evaluating cloud_density at each voxel center instead
        # of being read from the scene file. That means every downstream
        # consumer (delta tracking, local majorants, the GPU upload path,
        # NEE ratio tracking) needs zero changes: they already only know
        # "dense grid with a density array", not how it was produced.
        #
        # PBRT-v4 defaults sigma_a/sigma_s to ConstantSpectrum(1) when
        # unspecified, same as GridMedium/NanoVDBMedium above.
        var c_density = params.get_float("density", Float32(1))
        var c_wispiness = params.get_float("wispiness", Float32(1))
        var c_frequency = params.get_float("frequency", Float32(5))
        var sa_eff_c = sa if sa_set else RGB(Float32(1))
        var ss_eff_c = ss if ss_set else RGB(Float32(1))
        var name_str_c = String(unsafe_from_utf8_ptr=name_buf.as_imm())
        s[unsafe_offset=0].med_names.append(name_str_c)
        s[unsafe_offset=0].med_sa.append(sa_eff_c.r * scale); s[unsafe_offset=0].med_sa.append(sa_eff_c.g * scale); s[unsafe_offset=0].med_sa.append(sa_eff_c.b * scale)
        s[unsafe_offset=0].med_ss.append(ss_eff_c.r * scale); s[unsafe_offset=0].med_ss.append(ss_eff_c.g * scale); s[unsafe_offset=0].med_ss.append(ss_eff_c.b * scale)
        s[unsafe_offset=0].med_g.append(g_val)
        s[unsafe_offset=0].med_grid_idx.append(Int32(len(s[unsafe_offset=0].grid_nx)))
        s[unsafe_offset=0].med_nvdb_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_temp_idx.append(Int32(-1))
        s[unsafe_offset=0].med_le_scale.append(Float32(0)); s[unsafe_offset=0].med_temp_offset.append(Float32(0)); s[unsafe_offset=0].med_temp_scale.append(Float32(1))
        s[unsafe_offset=0].med_is_sss.append(Int32(0))

        s[unsafe_offset=0].grid_nx.append(Int32(CLOUD_BAKE_RES)); s[unsafe_offset=0].grid_ny.append(Int32(CLOUD_BAKE_RES)); s[unsafe_offset=0].grid_nz.append(Int32(CLOUD_BAKE_RES))
        s[unsafe_offset=0].grid_p0.append(g_p0.r); s[unsafe_offset=0].grid_p0.append(g_p0.g); s[unsafe_offset=0].grid_p0.append(g_p0.b)
        s[unsafe_offset=0].grid_p1.append(g_p1.r); s[unsafe_offset=0].grid_p1.append(g_p1.g); s[unsafe_offset=0].grid_p1.append(g_p1.b)
        for ci in range(16):
            s[unsafe_offset=0].grid_ctm.append(s[unsafe_offset=0].ctm[ci])
        s[unsafe_offset=0].grid_density_base.append(Int32(len(s[unsafe_offset=0].grid_density)))

        # CORRECTED 2026-09-10 (second pass): the "auto-size p0/p1 to the
        # enclosing shape's AABB" heuristic below (formerly gated on
        # `has_explicit_bounds`, now removed) was itself wrong, verified
        # against real pbrt source (media.h). pbrt's CloudMedium::Density(p)
        # never references `bounds` in its body -- bounds exists PURELY to
        # gate SampleRay's ray-medium overlap test. A ray segment outside
        # [p0,p1] finds an EMPTY majorant iterator and is treated as vacuum
        # -- Density() is simply never called there. So when a scene omits
        # p0/p1 (as clouds.pbrt does), pbrt's own default of a 1-unit box
        # `(0,0,0)-(1,1,1)` is NOT "most of a several-unit medium clipped to
        # zero density" (the previous framing) -- it is pbrt's actual,
        # intended behavior: the cloud is a small puffy region sitting in
        # one corner of whatever larger shape (here, a radius-1 sphere)
        # marks the medium interface, with the rest of that shape's volume
        # being real vacuum. Verified directly against clouds.pbrt: no
        # transform between WorldBegin and MakeNamedMedium "c", so the
        # medium's own default box IS world-space [0,1]^3 -- entirely
        # inside the sphere (centered (.5,.5,.5), radius 1), occupying only
        # ~24% of its volume.
        #
        # Expanding the baked box to the sphere's AABB [-.5,1.5]^3 (8x the
        # volume) evaluated cloud_density() far outside the altitude
        # falloff's intended y in [0,1] range. For y<0 specifically, the
        # additive base-pad term `2*max(0, 0.5-p.y)` exceeds 1 and clamps
        # to full saturation (density=1, zero noise variation) across the
        # ENTIRE y<0 half of the baked volume -- a large solid-opaque
        # region with no structure at all, which is what was reading as
        # "low contrast" (StdDev ~0.09 vs real pbrt's ~0.24 on this scene).
        #
        # Fix: just bake into [p0,p1] directly, exactly as given (pbrt's
        # own defaults if the scene omits them) -- matching pbrt exactly,
        # same as the sibling "uniformgrid" branch just above already does
        # for its own p0/p1. No shape-AABB inference, no deferred/pending
        # bake. `grid_cloud_pending` and `_resolve_pending_cloud_grid`
        # become permanently dead (every grid_cloud_pending entry is now
        # always 0) -- left in place rather than deleted, since
        # _resolve_pending_cloud_grid's own early-return on pending==0
        # already makes every call a safe no-op, and removing the dead
        # code is a separate, lower-risk cleanup than this fix.
        s[unsafe_offset=0].grid_cloud_pending.append(Int32(0))
        s[unsafe_offset=0].grid_cloud_density.append(Float32(0)); s[unsafe_offset=0].grid_cloud_wispiness.append(Float32(0)); s[unsafe_offset=0].grid_cloud_frequency.append(Float32(0))
        # Voxel centers at half-integer offsets within [p0,p1] -- matches
        # grid_sample_density's own half-texel convention (geometry.mojo),
        # so the trilinear reconstruction of this baked grid lands exactly
        # where cloud_density was evaluated, not off by half a cell.
        var perm = _perlin_perm_table()
        var ext_x = g_p1.r - g_p0.r
        var ext_y = g_p1.g - g_p0.g
        var ext_z = g_p1.b - g_p0.b
        var inv_res = Float32(1.0) / Float32(CLOUD_BAKE_RES)
        for zi in range(CLOUD_BAKE_RES):
            var pz = g_p0.b + ext_z * (Float32(zi) + Float32(0.5)) * inv_res
            for yi in range(CLOUD_BAKE_RES):
                var py = g_p0.g + ext_y * (Float32(yi) + Float32(0.5)) * inv_res
                for xi in range(CLOUD_BAKE_RES):
                    var px = g_p0.r + ext_x * (Float32(xi) + Float32(0.5)) * inv_res
                    var dv = cloud_density(perm, px, py, pz, c_frequency, c_wispiness, c_density)
                    s[unsafe_offset=0].grid_density.append(dv)
    elif is_nvdb:
        # PBRT-v4's NanoVDBMedium ALSO defaults sigma_a/sigma_s to
        # ConstantSpectrum(1) when unspecified (media.cpp), same as
        # GridMedium above -- see that branch's comment.
        var sa_eff_v = sa if sa_set else RGB(Float32(1))
        var ss_eff_v = ss if ss_set else RGB(Float32(1))
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_imm())
        s[unsafe_offset=0].med_names.append(name_str)
        s[unsafe_offset=0].med_sa.append(sa_eff_v.r * scale); s[unsafe_offset=0].med_sa.append(sa_eff_v.g * scale); s[unsafe_offset=0].med_sa.append(sa_eff_v.b * scale)
        s[unsafe_offset=0].med_ss.append(ss_eff_v.r * scale); s[unsafe_offset=0].med_ss.append(ss_eff_v.g * scale); s[unsafe_offset=0].med_ss.append(ss_eff_v.b * scale)
        s[unsafe_offset=0].med_g.append(g_val)
        s[unsafe_offset=0].med_grid_idx.append(Int32(-1))
        s[unsafe_offset=0].med_nvdb_idx.append(Int32(len(s[unsafe_offset=0].nvdb_filenames)))

        # The actual .nvdb file is loaded later, at finalize_scene time (the
        # C bridge call needs a real file path, and this is the only place
        # that already resolves scene-relative paths against s[0].scene_dir
        # -- do that resolution here, at parse time, same as every other
        # filename param, but defer the actual load).
        var nvdb_rel = params.get_string("filename", "")
        var nvdb_full = scene_path(s[unsafe_offset=0].scene_dir, nvdb_rel, "NanoVDB medium")
        var density_name = params.get_string("densityname", "density")
        s[unsafe_offset=0].nvdb_filenames.append(nvdb_full)
        s[unsafe_offset=0].nvdb_gridnames.append(density_name)
        for ci in range(16):
            s[unsafe_offset=0].nvdb_ctm.append(s[unsafe_offset=0].ctm[ci])

        # Emissive volume (pbrt NanoVDBMedium): a SECOND grid, "temperature",
        # in the SAME file. Always registered -- if the file has no such grid
        # the load simply yields an empty grid whose index bbox rejects every
        # lookup, so Le evaluates to 0 and the medium is non-emissive with no
        # special-casing anywhere downstream. "temperatureoffset" falls back to
        # "temperaturecutoff", matching pbrt's own parameter aliasing.
        var le_scale = params.get_float("Lescale", Float32(1))
        var temp_cut = params.get_float("temperaturecutoff", Float32(0))
        var temp_off = params.get_float("temperatureoffset", temp_cut)
        var temp_scl = params.get_float("temperaturescale", Float32(1))
        var temp_name = params.get_string("temperaturename", "temperature")
        s[unsafe_offset=0].med_nvdb_temp_idx.append(Int32(len(s[unsafe_offset=0].nvdb_filenames)))
        s[unsafe_offset=0].med_le_scale.append(le_scale)
        s[unsafe_offset=0].med_temp_offset.append(temp_off)
        s[unsafe_offset=0].med_temp_scale.append(temp_scl)
        s[unsafe_offset=0].med_is_sss.append(Int32(0))
        s[unsafe_offset=0].nvdb_filenames.append(nvdb_full)
        s[unsafe_offset=0].nvdb_gridnames.append(temp_name)
        for ci in range(16):
            s[unsafe_offset=0].nvdb_ctm.append(s[unsafe_offset=0].ctm[ci])
    else:
        # Unsupported medium type. This branch used not to exist, so an
        # unrecognised type appended NOTHING to med_names -- lookup_medium
        # then returned -1 and the MediumInterface bound to no medium at
        # all, silently -- `clouds.pbrt` (type "cloud") used to hit exactly
        # this (now fixed, see the is_cloud branch above): a bare
        # null-material sphere you see straight through, no cloud, no
        # warning. Same defect class as the .spd/.ply.gz/scale-texture
        # drops -- parsed, recognised as "not mine", and discarded without
        # a word.
        var bad_name = String(unsafe_from_utf8_ptr=name_buf.as_imm())
        warn_unsupported_in("medium type", type_str, "medium", bad_name,
                            "it is DROPPED, so any MediumInterface naming it renders as empty space",
                            "homogeneous, uniformgrid, nanovdb, cloud")
    name_buf.unsafe_free()

def lookup_medium(s: Pointer[SceneParseState, MutUntrackedOrigin],
                  name: Pointer[UInt8, MutUntrackedOrigin]) -> Int32:
    if name[unsafe_offset=0] == UInt8(0):
        return Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=name.as_imm())
    for i in range(len(s[unsafe_offset=0].med_names)):
        if s[unsafe_offset=0].med_names[i] == name_str:
            return Int32(i)
    return Int32(-1)

def handle_medium_interface(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                            s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var inside_buf  = unsafe_alloc[UInt8](64)
    var outside_buf = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, inside_buf, 64)
    _ = scanner_parse_quoted_string(handle, outside_buf, 64)
    s[unsafe_offset=0].cur_attr.inside_medium  = lookup_medium(s, inside_buf)
    s[unsafe_offset=0].cur_attr.outside_medium = lookup_medium(s, outside_buf)
    inside_buf.unsafe_free(); outside_buf.unsafe_free()

# ── Shape handlers ────────────────────────────────────────────────────────────

def handle_sphere_shape(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                             s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var params = _psc_collect_params(handle)
    var radius = params.get_float("radius", Float32(1.0))

    var cx = s[unsafe_offset=0].ctm[12]
    var cy = s[unsafe_offset=0].ctm[13]
    var cz = s[unsafe_offset=0].ctm[14]
    var sx = sqrt(s[unsafe_offset=0].ctm[0]*s[unsafe_offset=0].ctm[0] + s[unsafe_offset=0].ctm[1]*s[unsafe_offset=0].ctm[1] + s[unsafe_offset=0].ctm[2]*s[unsafe_offset=0].ctm[2])
    if sx < Float32(1e-6): sx = Float32(1.0)
    radius *= sx
    s[unsafe_offset=0].spheres_cx.append(cx)
    s[unsafe_offset=0].spheres_cy.append(cy)
    s[unsafe_offset=0].spheres_cz.append(cz)
    s[unsafe_offset=0].spheres_r.append(radius)
    s[unsafe_offset=0].spheres_mat.append(s[unsafe_offset=0].cur_attr.mat_idx)
    s[unsafe_offset=0].spheres_inside_med.append(s[unsafe_offset=0].cur_attr.inside_medium)

    # If this sphere is the inside boundary of a "cloud" medium whose scene
    # file never gave it explicit p0/p1 bounds, size the medium's baked
    # density grid to this sphere's own world AABB now that we know it --
    # see _resolve_pending_cloud_grid's docstring. Only the FIRST shape
    # found using a given pending cloud medium resolves it (documented
    # scope: multiple shapes sharing one unbounded cloud medium is not
    # handled -- rare, and each would want its own bounds anyway).
    var inside_idx = s[unsafe_offset=0].cur_attr.inside_medium
    if inside_idx >= Int32(0) and Int(inside_idx) < len(s[unsafe_offset=0].med_grid_idx):
        var g_idx = s[unsafe_offset=0].med_grid_idx[Int(inside_idx)]
        if g_idx >= Int32(0):
            _resolve_pending_cloud_grid(s, g_idx,
                cx - radius, cy - radius, cz - radius,
                cx + radius, cy + radius, cz + radius)
    s[unsafe_offset=0].spheres_outside_med.append(s[unsafe_offset=0].cur_attr.outside_medium)
    s[unsafe_offset=0].spheres_al.append(s[unsafe_offset=0].cur_attr.is_alight)
    s[unsafe_offset=0].spheres_rgb.append(s[unsafe_offset=0].cur_attr.al_rgb)

def _resolve_pending_cloud_grid(
    s: Pointer[SceneParseState, MutUntrackedOrigin], grid_idx: Int32,
    min_x: Float32, min_y: Float32, min_z: Float32,
    max_x: Float32, max_y: Float32, max_z: Float32,
):
    """Rebakes a "cloud" medium's density grid (see handle_named_medium's
    is_cloud branch) once the first shape using it as an inside-interface is
    known, sizing p0/p1 to that shape's world AABB instead of the synthetic
    default -- called from handle_sphere_shape et al. No-op if grid_idx is
    invalid or this grid isn't a pending cloud (already resolved, or never
    was one -- e.g. a plain "uniformgrid" medium)."""
    if grid_idx < Int32(0) or Int(grid_idx) >= len(s[unsafe_offset=0].grid_cloud_pending):
        return
    var gi = Int(grid_idx)
    if s[unsafe_offset=0].grid_cloud_pending[gi] == Int32(0):
        return
    s[unsafe_offset=0].grid_cloud_pending[gi] = Int32(0)
    s[unsafe_offset=0].grid_p0[gi * 3 + 0] = min_x; s[unsafe_offset=0].grid_p0[gi * 3 + 1] = min_y; s[unsafe_offset=0].grid_p0[gi * 3 + 2] = min_z
    s[unsafe_offset=0].grid_p1[gi * 3 + 0] = max_x; s[unsafe_offset=0].grid_p1[gi * 3 + 1] = max_y; s[unsafe_offset=0].grid_p1[gi * 3 + 2] = max_z

    var c_density = s[unsafe_offset=0].grid_cloud_density[gi]
    var c_wispiness = s[unsafe_offset=0].grid_cloud_wispiness[gi]
    var c_frequency = s[unsafe_offset=0].grid_cloud_frequency[gi]
    var perm = _perlin_perm_table()
    var ext_x = max_x - min_x
    var ext_y = max_y - min_y
    var ext_z = max_z - min_z
    var inv_res = Float32(1.0) / Float32(CLOUD_BAKE_RES)
    var base = Int(s[unsafe_offset=0].grid_density_base[gi])
    var idx = base
    for zi in range(CLOUD_BAKE_RES):
        var pz = min_z + ext_z * (Float32(zi) + Float32(0.5)) * inv_res
        for yi in range(CLOUD_BAKE_RES):
            var py = min_y + ext_y * (Float32(yi) + Float32(0.5)) * inv_res
            for xi in range(CLOUD_BAKE_RES):
                var px = min_x + ext_x * (Float32(xi) + Float32(0.5)) * inv_res
                var dv = cloud_density(perm, px, py, pz, c_frequency, c_wispiness, c_density)
                s[unsafe_offset=0].grid_density[idx] = dv
                idx += 1

comptime DISK_TESSELLATION_SEGMENTS: Int = 32

def handle_disk_shape(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                           s: Pointer[SceneParseState, MutUntrackedOrigin]):
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

def handle_bilinearmesh_shape(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                               s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """PBRT `Shape "bilinearmesh"`: one or more planar-quad patches given as
    "point3 P" (+ optional "integer indices", 4 per patch; pbrt's own
    default when omitted and len(P)==4 is a single patch {0,1,2,3} -- see
    pbrt-v4 BilinearPatch::CreateMesh). Each patch's 4 control points are in
    pbrt's (p00, p10, p01, p11) convention -- verified against
    ~/src/pbrt-v4/src/pbrt/shapes.cpp's BilinearPatch::NormalBounds
    (v[0]=p00, v[1]=p10, v[2]=p01, v[3]=p11; n00 = cross(p10-p00, p01-p00)).

    gonzales has no native bilinear-patch primitive (and the general
    doubly-ruled surface is curved for a non-planar quad, which the
    triangle-fan approximation below does not reproduce) -- but every known
    use in the corpus (sportscar's area-light emitter planes) is a flat
    quad, where two triangles are exact, not an approximation. Tessellated
    into a mesh at parse time exactly like disk/loopsubdiv above, so it goes
    through store_mesh and gets full trianglemesh machinery (BVH, area-light
    NEE/ReSTIR, GPU, ...) for free.

    Winding (p00, p10, p01) / (p10, p11, p01) matches pbrt's own outward
    normal at corner (0,0) -- both triangles share that same winding sense,
    so a planar quad's normal is uniform across the tessellation."""
    var params = _psc_collect_params(handle)
    var p = params.take_floats("P")
    var indices = params.take_ints("indices")
    var n_pts = Int32(len(p) // 3)

    if len(indices) == 0:
        if n_pts == Int32(4):
            indices.append(Int32(0)); indices.append(Int32(1))
            indices.append(Int32(2)); indices.append(Int32(3))
        else:
            print("Warning: Shape \"bilinearmesh\" with no \"integer indices\" needs exactly 4 points per pbrt's own single-patch default; got", n_pts, "-- shape dropped.")
            return
    elif len(indices) % 4 != 0:
        print("Warning: Shape \"bilinearmesh\" \"integer indices\" length", len(indices), "is not a multiple of 4 -- shape dropped.")
        return

    var n_patches = len(indices) // 4
    var idx = List[Int32]()
    for patch in range(n_patches):
        var i00 = indices[patch * 4 + 0]
        var i10 = indices[patch * 4 + 1]
        var i01 = indices[patch * 4 + 2]
        var i11 = indices[patch * 4 + 3]
        if i00 >= n_pts or i10 >= n_pts or i01 >= n_pts or i11 >= n_pts or i00 < 0 or i10 < 0 or i01 < 0 or i11 < 0:
            print("Warning: Shape \"bilinearmesh\" patch", patch, "indexes past the end of \"P\" -- shape dropped.")
            return
        idx.append(i00); idx.append(i10); idx.append(i01)
        idx.append(i10); idx.append(i11); idx.append(i01)

    var n_tris = Int32(len(idx) // 3)
    store_mesh(s, p.unsafe_ptr(), idx.unsafe_ptr(), n_pts, n_tris)

def handle_shape(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                     s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var shape_type = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, shape_type, 64)

    var is_tri = _psc_streq(shape_type, "trianglemesh")
    var is_ply = _psc_streq(shape_type, "plymesh")
    var is_curve = _psc_streq(shape_type, "curve")
    var is_sphere = _psc_streq(shape_type, "sphere")
    var is_loopsubdiv = _psc_streq(shape_type, "loopsubdiv")
    var is_disk = _psc_streq(shape_type, "disk")
    var is_bilinearmesh = _psc_streq(shape_type, "bilinearmesh")
    var shape_type_name = String(unsafe_from_utf8_ptr=shape_type.as_imm())
    shape_type.unsafe_free()

    if is_disk:
        # Tessellated into a mesh (handle_disk_shape above), so it goes
        # through store_mesh exactly like trianglemesh/loopsubdiv --
        # already usable inside ObjectBegin/ObjectEnd instancing templates,
        # no object_depth restriction needed (unlike curve/sphere above,
        # which are native, uninstanced primitives).
        handle_disk_shape(handle, s)
        return

    if is_bilinearmesh:
        # Tessellated into a mesh (handle_bilinearmesh_shape above), same
        # store_mesh path as disk/loopsubdiv -- usable inside
        # ObjectBegin/ObjectEnd instancing templates.
        handle_bilinearmesh_shape(handle, s)
        return

    if is_curve:
        if s[unsafe_offset=0].object_depth == 0:
            handle_curve_shape(handle, s)
        else:
            # Curve instancing inside ObjectBegin/ObjectEnd isn't supported yet
            # (only trianglemesh/plymesh templates are, see _psc_finish_object_def) —
            # skip rather than silently adding an un-instanced curve at the
            # template's definition-space position.
            _psc_skip_params(handle)
        return

    if is_sphere:
        if s[unsafe_offset=0].object_depth == 0:
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
        # Every shape this parser knows returned above, so this one is
        # unknown. It used to be skipped without a word, which is the silent
        # drop this codebase keeps paying for: the geometry is simply absent
        # from an otherwise plausible image.
        warn_unsupported("shape type", shape_type_name, "the shape is skipped, leaving a hole in the scene",
                         "trianglemesh, plymesh, loopsubdiv, disk, bilinearmesh, sphere, curve")
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

        var full_path = unsafe_alloc[UInt8](PSC_FILE_MAX * 2)
        var ply_path = scene_path(s[unsafe_offset=0].scene_dir, ply_filename_str, "PLY mesh")
        var fn_bytes = ply_path.unsafe_ptr()
        var fn_len = ply_path.byte_length()
        var fn_i = 0
        while fn_i < fn_len and fn_i < PSC_FILE_MAX * 2 - 1:
            full_path[unsafe_offset=fn_i] = fn_bytes[unsafe_offset=fn_i]
            fn_i += 1
        full_path[unsafe_offset=fn_i] = UInt8(0)

        var ply_pts     = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
        var ply_nv      = unsafe_alloc[Int32](1)
        var ply_idx     = unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](1)
        var ply_nt      = unsafe_alloc[Int32](1)
        var ply_uvs     = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
        var ply_has_uvs = unsafe_alloc[Int32](1)
        var ply_nrm     = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
        var ply_has_nrm = unsafe_alloc[Int32](1)
        ply_uvs[unsafe_offset=0] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        ply_has_uvs[unsafe_offset=0] = Int32(0)
        ply_nrm[unsafe_offset=0] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        ply_has_nrm[unsafe_offset=0] = Int32(0)
        # For .ply.gz, use the decompressed .ply sibling (strip ".gz"),
        # auto-decompressing once if it isn't there yet. load_ply reads the
        # file raw and rejects anything whose first word isn't "ply", so
        # handing it gzip bytes fails outright -- pbrt-v4-scenes ships
        # ganesha, sssdragon and lte-orb's meshes gzipped, and without this
        # their geometry silently vanished from the render (the statue, the
        # dragon, three of lte-orb's four meshes). Mirrors the `.pbrt.gz`
        # include path below, which already decompresses this way.
        var fp_len = 0
        while full_path[unsafe_offset=fp_len] != UInt8(0): fp_len += 1
        var ends_gz = (fp_len >= 4 and
                       full_path[unsafe_offset=fp_len-3] == UInt8(46) and
                       full_path[unsafe_offset=fp_len-2] == UInt8(103) and
                       full_path[unsafe_offset=fp_len-1] == UInt8(122))
        var ok = Int32(0)
        if ends_gz:
            var ap = unsafe_alloc[UInt8](fp_len - 2)
            for ci in range(fp_len - 3): ap[unsafe_offset=ci] = full_path[unsafe_offset=ci]
            ap[unsafe_offset=fp_len - 3] = UInt8(0)
            var ap_str = String(unsafe_from_utf8_ptr=ap.as_imm())
            if not exists(ap_str):
                var gz_str = String(unsafe_from_utf8_ptr=full_path.as_imm())
                print("decompressing", gz_str, "(one-time, cached alongside it)")
                try:
                    _ = run("gzip -dk '" + gz_str + "'")
                except:
                    pass
                if not exists(ap_str):
                    print("PLY gunzip FAILED (is `gzip` installed?):", gz_str)
            ok = load_ply(ap, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
            ap.unsafe_free()
        if ok == 0:
            ok = load_ply(full_path, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
        if ok == 0:
            print("PLY load FAILED:", String(unsafe_from_utf8_ptr=full_path.as_imm()))
            full_path.unsafe_free()
            ply_pts.unsafe_free(); ply_nv.unsafe_free(); ply_idx.unsafe_free(); ply_nt.unsafe_free()
            ply_uvs.unsafe_free(); ply_has_uvs.unsafe_free(); ply_nrm.unsafe_free(); ply_has_nrm.unsafe_free()
            return
        full_path.unsafe_free()
        var nv = ply_nv[unsafe_offset=0]
        var nt = ply_nt[unsafe_offset=0]
        if nv <= 0 or nt <= 0:
            ply_pts[unsafe_offset=0].unsafe_free(); ply_idx[unsafe_offset=0].unsafe_free()
            if ply_has_uvs[unsafe_offset=0] != 0:
                ply_uvs[unsafe_offset=0].unsafe_free()
            if ply_has_nrm[unsafe_offset=0] != 0:
                ply_nrm[unsafe_offset=0].unsafe_free()
            ply_pts.unsafe_free(); ply_nv.unsafe_free(); ply_idx.unsafe_free(); ply_nt.unsafe_free()
            ply_uvs.unsafe_free(); ply_has_uvs.unsafe_free(); ply_nrm.unsafe_free(); ply_has_nrm.unsafe_free()
            return
        var tmp_f2 = ply_pts[unsafe_offset=0]
        var tmp_i2 = ply_idx[unsafe_offset=0]
        store_mesh(s, tmp_f2, tmp_i2, nv, nt)
        if ply_has_uvs[unsafe_offset=0] != 0:
            var uv_ptr = ply_uvs[unsafe_offset=0]
            var n_uv_floats = Int(nv) * 2
            for uvi in range(n_uv_floats):
                s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1].uvs.append(uv_ptr[unsafe_offset=uvi])
            uv_ptr.unsafe_free()
        if ply_has_nrm[unsafe_offset=0] != 0:
            var nrm_ptr = ply_nrm[unsafe_offset=0]
            var ctm_inv = unsafe_alloc[Float32](16)
            _ = matrix_invert(s[unsafe_offset=0].ctm.unsafe_ptr(), ctm_inv)
            var nrm_world = unsafe_alloc[Float32](Int(nv) * 3)
            transform_normals(ctm_inv, nrm_ptr, nv, nrm_world)
            ref last_mesh = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
            last_mesh.normals.reserve(Int(nv) * 3)
            for ni in range(Int(nv)):
                var nx = nrm_world[unsafe_offset=ni*3+0]; var ny = nrm_world[unsafe_offset=ni*3+1]; var nz = nrm_world[unsafe_offset=ni*3+2]
                var nlen = sqrt(nx*nx + ny*ny + nz*nz)
                if nlen > Float32(1e-12):
                    var inv = Float32(1.0) / nlen
                    nx *= inv; ny *= inv; nz *= inv
                last_mesh.normals.append(nx)
                last_mesh.normals.append(ny)
                last_mesh.normals.append(nz)
            nrm_world.unsafe_free(); ctm_inv.unsafe_free(); nrm_ptr.unsafe_free()
        tmp_f2.unsafe_free(); tmp_i2.unsafe_free()
        ply_pts.unsafe_free(); ply_nv.unsafe_free(); ply_idx.unsafe_free(); ply_nt.unsafe_free()
        ply_uvs.unsafe_free(); ply_has_uvs.unsafe_free(); ply_nrm.unsafe_free(); ply_has_nrm.unsafe_free()
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
    var n_list = params.take_floats("N")

    var n_verts = Int32(len(p_list) / 3)
    var n_tris  = Int32(len(i_list) / 3)

    if n_verts <= 0 or n_tris <= 0:
        return

    store_mesh(s, p_list.unsafe_ptr(), i_list.unsafe_ptr(), n_verts, n_tris)
    if Int32(len(uv_list)) >= n_verts * Int32(2):
        for ui in range(Int(n_verts) * 2):
            s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1].uvs.append(uv_list[ui])
    # "N" (per-vertex shading normals) -- same treatment plymesh already
    # gives its PLY-supplied normals: inverse-transpose CTM into world
    # space, then normalize. Dropping these (as this handler did) is not
    # merely a loss of smooth shading: a one-sided area light's facing test
    # consults them, and pbrt takes them as authoritative over the index
    # winding (Triangle::InteractionFromIntersection does
    # `n = FaceForward(n, ns)`). staircase2's big window emitter supplies
    # N = (1,0,0) against indices that wind to (-1,0,0), so with N dropped
    # it faced away from the room: rendered black where the reference shows
    # its full 4.575/3.591/1.550 radiance, and lit nothing through NEE.
    if Int32(len(n_list)) >= n_verts * Int32(3):
        var ctm_inv = unsafe_alloc[Float32](16)
        _ = matrix_invert(s[unsafe_offset=0].ctm.unsafe_ptr(), ctm_inv)
        var nrm_world = unsafe_alloc[Float32](Int(n_verts) * 3)
        var nrm_src = unsafe_alloc[Float32](Int(n_verts) * 3)
        for ni in range(Int(n_verts) * 3): nrm_src[unsafe_offset=ni] = n_list[ni]
        transform_normals(ctm_inv, nrm_src, n_verts, nrm_world)
        nrm_src.unsafe_free()
        ref nm = s[unsafe_offset=0].meshes[len(s[unsafe_offset=0].meshes) - 1]
        nm.normals.reserve(Int(n_verts) * 3)
        for ni in range(Int(n_verts)):
            var nx = nrm_world[unsafe_offset=ni*3+0]; var ny = nrm_world[unsafe_offset=ni*3+1]; var nz = nrm_world[unsafe_offset=ni*3+2]
            var nlen = sqrt(nx*nx + ny*ny + nz*nz)
            if nlen > Float32(1e-12):
                var inv = Float32(1.0) / nlen
                nx *= inv; ny *= inv; nz *= inv
            nm.normals.append(nx)
            nm.normals.append(ny)
            nm.normals.append(nz)
        nrm_world.unsafe_free(); ctm_inv.unsafe_free()

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

def _psc_get_sigma_or_rgb(params: ParameterDictionary, name: StringLiteral, default: RGB) -> RGB:
    """Medium "sigma_a"/"sigma_s" only -- NOT `_psc_get_float_or_rgb`'s
    texture-RGB duality, which this deliberately does not share (only 2
    call sites, both here).

    pbrt declares these `"spectrum sigma_a"`. `ParamValue` collapses every
    float-bearing pbrt type (float/rgb/spectrum-numeric-array/...) into one
    flat `floats` list with no surviving type tag (see its own docstring),
    so a bare `len(f)` count is the only signal left to tell an RGB TRIPLE
    (always exactly 3 floats: r, g, b) apart from a SPECTRUM NUMERIC ARRAY
    (an even, usually >3, count of alternating `wavelength value` pairs --
    e.g. `[200 .01 900 .01]` means sigma_a=.01 at both 200nm and 900nm, NOT
    the RGB triple (200, .01, 900)).

    Before this function existed, both medium call sites went through
    `_psc_get_float_or_rgb`, which has no spectrum case at all and read ANY
    `len(f) >= 3` as a bare RGB triple -- so `[200 .01 900 .01]` (4 floats)
    became `RGB(200, .01, 900)`: sigma_a in the R and B channels off by
    four orders of magnitude (200/900 instead of .01), sigma_t there
    (R/B ~200-900 vs the real ~10) making those channels essentially fully
    opaque at any real path length, while G alone carried sane values --
    exactly the washed-out, low-contrast "doesn't look right" a cloud
    medium using this exact spectrum form produced (`clouds.pbrt`'s
    "spectrum sigma_s"/"spectrum sigma_a"). The existing comment above this
    function's own call sites already claimed the intended behaviour was
    "mean of samples, replicated to all 3 channels" -- this makes the code
    match that stated intent, using only the VALUE half of each pair (the
    odd indices), not the wavelengths themselves."""
    var f = params.get_floats(name)
    var n = len(f)
    if n == 1:
        return RGB(f[0])
    if n == 3:
        return RGB(f[0], f[1], f[2])
    if n >= 4 and n % 2 == 0:
        var sum_v = Float32(0.0)
        var n_pairs = n // 2
        for i in range(n_pairs):
            sum_v += f[2 * i + 1]
        var mean_v = sum_v / Float32(n_pairs)
        return RGB(mean_v)
    return default

def handle_texture(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
                       s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var tex_name = unsafe_alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, tex_name, PSC_NAME_MAX)
    var tex_type = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_type, 64)
    var tex_class = unsafe_alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_class, 64)
    var name_str = String(unsafe_from_utf8_ptr=tex_name.as_imm())
    tex_name.unsafe_free()

    if _psc_streq(tex_class, "constant"):
        tex_type.unsafe_free(); tex_class.unsafe_free()
        var params = _psc_collect_params(handle)
        var crgb = _psc_get_float_or_rgb(params, "value", RGB(Float32(0.5)))
        s[unsafe_offset=0].const_tex_names.append(name_str)
        s[unsafe_offset=0].const_tex_rgb.append(crgb.r)
        s[unsafe_offset=0].const_tex_rgb.append(crgb.g)
        s[unsafe_offset=0].const_tex_rgb.append(crgb.b)
        return
    if _psc_streq(tex_class, "checkerboard"):
        tex_type.unsafe_free(); tex_class.unsafe_free()
        var params = _psc_collect_params(handle)
        # pbrt defaults: tex1=1 (white), tex2=0 (black), uscale=vscale=1.
        var ktex1 = _psc_get_float_or_rgb(params, "tex1", RGB(Float32(1.0)))
        var ktex2 = _psc_get_float_or_rgb(params, "tex2", RGB(Float32(0.0)))
        var kuscale = params.get_float("uscale", Float32(1.0))
        var kvscale = params.get_float("vscale", Float32(1.0))
        s[unsafe_offset=0].checker_tex_names.append(name_str)
        s[unsafe_offset=0].checker_tex1.append(ktex1.r); s[unsafe_offset=0].checker_tex1.append(ktex1.g); s[unsafe_offset=0].checker_tex1.append(ktex1.b)
        s[unsafe_offset=0].checker_tex2.append(ktex2.r); s[unsafe_offset=0].checker_tex2.append(ktex2.g); s[unsafe_offset=0].checker_tex2.append(ktex2.b)
        s[unsafe_offset=0].checker_uscale.append(kuscale)
        s[unsafe_offset=0].checker_vscale.append(kvscale)
        return
    if _psc_streq(tex_class, "scale"):
        tex_type.unsafe_free(); tex_class.unsafe_free()
        var params = _psc_collect_params(handle)
        # Both operands may be a nested texture or a literal -- see the
        # scale_tex_* comment in parse_types.mojo. get_string returns "" when
        # the operand has no string form, which is the "it's a literal" marker.
        var base_name = params.get_string("tex", "")
        var base_rgb = _psc_get_float_or_rgb(params, "tex", RGB(Float32(1.0)))
        var scale_name = params.get_string("scale", "")
        var scale_val = params.get_float("scale", Float32(1))
        s[unsafe_offset=0].scale_tex_names.append(name_str)
        s[unsafe_offset=0].scale_tex_base.append(base_name)
        s[unsafe_offset=0].scale_tex_base_rgb.append(base_rgb.r)
        s[unsafe_offset=0].scale_tex_base_rgb.append(base_rgb.g)
        s[unsafe_offset=0].scale_tex_base_rgb.append(base_rgb.b)
        s[unsafe_offset=0].scale_tex_scale.append(scale_val)
        s[unsafe_offset=0].scale_tex_scale_name.append(scale_name)
        return
    if _psc_streq(tex_class, "mix"):
        tex_type.unsafe_free(); tex_class.unsafe_free()
        var params = _psc_collect_params(handle)
        # Each slot is either a nested texture reference (lands in the
        # dictionary's `strs` -- `"texture tex1" "name"`) or a literal
        # (lands in `floats` -- `"rgb tex1" [r g b]` / `"float tex1" [v]`).
        # get_string returns "" when the slot has no string form, which is
        # exactly the "this one is a constant" marker the tables expect.
        var t1_name = params.get_string("tex1", "")
        var t2_name = params.get_string("tex2", "")
        var am_name = params.get_string("amount", "")
        # pbrt's own defaults: tex1 = 0, tex2 = 1, amount = 0.5.
        var t1_rgb = _psc_get_float_or_rgb(params, "tex1", RGB(Float32(0.0)))
        var t2_rgb = _psc_get_float_or_rgb(params, "tex2", RGB(Float32(1.0)))
        var am_val = params.get_float("amount", Float32(0.5))
        s[unsafe_offset=0].mix_tex_names.append(name_str)
        s[unsafe_offset=0].mix_tex1_name.append(t1_name)
        s[unsafe_offset=0].mix_tex1_rgb.append(t1_rgb.r); s[unsafe_offset=0].mix_tex1_rgb.append(t1_rgb.g); s[unsafe_offset=0].mix_tex1_rgb.append(t1_rgb.b)
        s[unsafe_offset=0].mix_tex2_name.append(t2_name)
        s[unsafe_offset=0].mix_tex2_rgb.append(t2_rgb.r); s[unsafe_offset=0].mix_tex2_rgb.append(t2_rgb.g); s[unsafe_offset=0].mix_tex2_rgb.append(t2_rgb.b)
        s[unsafe_offset=0].mix_amount_name.append(am_name)
        s[unsafe_offset=0].mix_amount_val.append(am_val)
        return

    if not _psc_streq(tex_class, "imagemap"):
        # Unsupported texture class. Warn rather than dropping it in silence:
        # a silently-ignored texture renders as a plausible flat surface with
        # no error, which is exactly how the "scale"-on-reflectance gap
        # survived (killeroos' floor grid). Same convention as the
        # unsupported-material warnings in material_builder.mojo.
        var class_str = String(unsafe_from_utf8_ptr=tex_class.as_imm())
        var type_str  = String(unsafe_from_utf8_ptr=tex_type.as_imm())
        warn_unsupported_in("texture class", class_str + " (" + type_str + ")",
                            "texture", name_str, "it renders as a flat default",
                            "imagemap, scale, mix, checkerboard, constant")
        tex_type.unsafe_free(); tex_class.unsafe_free()
        _psc_skip_params(handle)
        return

    tex_type.unsafe_free(); tex_class.unsafe_free()
    var params = _psc_collect_params(handle)
    var filename = params.get_string("filename", "")
    if filename != "":
        var file_str = scene_path(s[unsafe_offset=0].scene_dir, filename, "image texture")
        s[unsafe_offset=0].tex_names.append(name_str)
        s[unsafe_offset=0].tex_files.append(file_str)

# ── ObjectBegin/ObjectEnd/ObjectInstance (two-level BVH instancing) ──────────
# See geometry.mojo's Instance_C docs and bvh.mojo's traverse_bvh2_core
# type==6 branch for the traversal side. Design: geometry inside
# ObjectBegin/ObjectEnd is parsed normally (baked at whatever CTM is active
# during that block, "definition space") but tagged is_object_template=True so
# finalize_scene excludes it from the ordinary top-level primitive list —
# instead a private BLAS is built once per template, and each ObjectInstance
# placement contributes a small TLAS leaf (transform + BLAS reference) rather
# than a duplicated copy of the geometry.

def _psc_finish_object_def(s: Pointer[SceneParseState, MutUntrackedOrigin]):
    """Called when the outermost ObjectEnd closes a template: mark the
    meshes captured since the matching ObjectBegin as template-only and
    record the template's (name, mesh range, definition-time CTM) for
    finalize_scene and later ObjectInstance directives."""
    var start = Int(s[unsafe_offset=0].pending_object_start)
    var end   = len(s[unsafe_offset=0].meshes)
    if end <= start:
        return  # empty object (e.g. only unsupported AreaLightSource/curves) — nothing to instance
    for i in range(start, end):
        s[unsafe_offset=0].meshes[i].is_object_template = True
    s[unsafe_offset=0].object_names.append(s[unsafe_offset=0].pending_object_name)
    s[unsafe_offset=0].object_mesh_start.append(Int32(start))
    s[unsafe_offset=0].object_mesh_end.append(Int32(end))
    for ci in range(16):
        s[unsafe_offset=0].object_ctm.append(s[unsafe_offset=0].pending_object_ctm[ci])

def _psc_emit_object_instance(s: Pointer[SceneParseState, MutUntrackedOrigin], name: String):
    """Called on ObjectInstance "name": look up the named template and record
    a placement (template index + obj_to_world/world_to_obj transforms,
    derived from the CTM active now vs. the CTM active at that template's
    ObjectBegin) for finalize_scene to turn into an Instance_C."""
    var tmpl_idx = -1
    for i in range(len(s[unsafe_offset=0].object_names)):
        if s[unsafe_offset=0].object_names[i] == name:
            tmpl_idx = i
            break
    if tmpl_idx < 0:
        return  # unknown/empty object name — nothing to place (e.g. all-skipped-content object)

    # obj_to_world = CTM_now * inverse(CTM_at_ObjectBegin) — since template
    # geometry is already baked in "CTM_at_ObjectBegin space", this maps it
    # into this placement's world position without re-parsing/duplicating it.
    var mdef = unsafe_alloc[Float32](16)
    for ci in range(16): mdef[unsafe_offset=ci] = s[unsafe_offset=0].object_ctm[tmpl_idx * 16 + ci]
    var mdef_inv = unsafe_alloc[Float32](16)
    _ = matrix_invert(mdef, mdef_inv)
    var obj_to_world = unsafe_alloc[Float32](16)
    matrix_multiply(s[unsafe_offset=0].ctm.unsafe_ptr(), mdef_inv, obj_to_world)
    var world_to_obj = unsafe_alloc[Float32](16)
    _ = matrix_invert(obj_to_world, world_to_obj)

    s[unsafe_offset=0].instance_template_idx.append(Int32(tmpl_idx))
    for ci in range(16):
        s[unsafe_offset=0].instance_obj_to_world.append(obj_to_world[unsafe_offset=ci])
        s[unsafe_offset=0].instance_world_to_obj.append(world_to_obj[unsafe_offset=ci])

    mdef.unsafe_free(); mdef_inv.unsafe_free(); obj_to_world.unsafe_free(); world_to_obj.unsafe_free()

# ── Main parse loop ───────────────────────────────────────────────────────────

def parse_scene_file(handle: Pointer[PbrtScanner, MutUntrackedOrigin],
              s: Pointer[SceneParseState, MutUntrackedOrigin]):
    var kw_buf = unsafe_alloc[UInt8](256)
    var ws_delims = unsafe_alloc[UInt8](4)
    ws_delims[unsafe_offset=0] = UInt8(32); ws_delims[unsafe_offset=1] = UInt8(9)
    ws_delims[unsafe_offset=2] = UInt8(10); ws_delims[unsafe_offset=3] = UInt8(13)

    while scanner_is_at_end(handle) == 0:
        var n = scanner_scan_token(handle, ws_delims, 4, kw_buf, 256)
        if n < 0:
            break
        if n == 0:
            continue

        if kw_buf[unsafe_offset=0] == UInt8(35):  # '#'
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
            var obj_name = unsafe_alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            if s[unsafe_offset=0].object_depth == 0:
                s[unsafe_offset=0].pending_object_name  = String(unsafe_from_utf8_ptr=obj_name.as_imm())
                s[unsafe_offset=0].pending_object_start = Int32(len(s[unsafe_offset=0].meshes))
                s[unsafe_offset=0].pending_object_ctm   = s[unsafe_offset=0].ctm.copy()
            obj_name.unsafe_free()
            s[unsafe_offset=0].object_depth += 1
        elif _psc_streq(kw_buf, "ObjectEnd"):
            if s[unsafe_offset=0].object_depth > 0:
                s[unsafe_offset=0].object_depth -= 1
                if s[unsafe_offset=0].object_depth == 0:
                    _psc_finish_object_def(s)
        elif _psc_streq(kw_buf, "ObjectInstance"):
            var obj_name = unsafe_alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            var inst_name = String(unsafe_from_utf8_ptr=obj_name.as_imm())
            obj_name.unsafe_free()
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
            ctm_push(s[unsafe_offset=0])
        elif _psc_streq(kw_buf, "TransformEnd"):
            ctm_pop(s[unsafe_offset=0])
        elif _psc_streq(kw_buf, "ReverseOrientation"):
            s[unsafe_offset=0].cur_attr.reverse_orient = not s[unsafe_offset=0].cur_attr.reverse_orient
        elif _psc_streq(kw_buf, "AreaLightSource"):
            if s[unsafe_offset=0].object_depth == 0:
                _psc_handle_area_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "LightSource"):
            if s[unsafe_offset=0].object_depth == 0:
                handle_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "Texture"):
            handle_texture(handle, s)
        elif _psc_streq(kw_buf, "Include") or _psc_streq(kw_buf, "Import"):
            var inc_name = unsafe_alloc[UInt8](PSC_FILE_MAX)
            _ = scanner_parse_quoted_string(handle, inc_name, PSC_FILE_MAX)
            var inc_path = unsafe_alloc[UInt8](PSC_FILE_MAX * 2)
            var inc_resolved = scene_path(s[unsafe_offset=0].scene_dir,
                                          String(unsafe_from_utf8_ptr=inc_name.as_imm()), "Include")
            var inc_bytes = inc_resolved.unsafe_ptr()
            var inc_len = inc_resolved.byte_length()
            var fi = 0
            while fi < inc_len and fi < PSC_FILE_MAX * 2 - 1:
                inc_path[unsafe_offset=fi] = inc_bytes[unsafe_offset=fi]
                fi += 1
            inc_path[unsafe_offset=fi] = UInt8(0)

            # `.pbrt.gz` includes (e.g. pbrt-v4-scenes' curve/hair geometry)
            # are NOT decompressed by the tokenizer -- without this, the
            # scanner opens the raw gzip bytes as if they were text and
            # silently tokenizes garbage, producing zero shapes with no
            # error at all. Auto-decompress once into a cached ".pbrt"
            # sibling next to the source (mirrors the pre-existing ".ply.gz"
            # sibling-file convention above), then open that instead.
            var open_path: Pointer[UInt8, MutUntrackedOrigin] = inc_path
            var inc_path_len = fi
            var ends_gz = (inc_path_len >= 3 and
                           inc_path[unsafe_offset=inc_path_len-3] == UInt8(46) and
                           inc_path[unsafe_offset=inc_path_len-2] == UInt8(103) and
                           inc_path[unsafe_offset=inc_path_len-1] == UInt8(122))
            var stripped = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
            if ends_gz:
                stripped = unsafe_alloc[UInt8](inc_path_len - 2)
                for ci in range(inc_path_len - 3):
                    stripped[unsafe_offset=ci] = inc_path[unsafe_offset=ci]
                stripped[unsafe_offset=inc_path_len - 3] = UInt8(0)
                var stripped_str = String(unsafe_from_utf8_ptr=stripped.as_imm())
                if not exists(stripped_str):
                    var inc_path_str = String(unsafe_from_utf8_ptr=inc_path.as_imm())
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
                var inc_len = Int(sub_handle[unsafe_offset=0].total_bytes)
                var rest_start = Int(handle[unsafe_offset=0].cursor)
                var rest_len = Int(handle[unsafe_offset=0].total_bytes) - rest_start
                var merged_len = inc_len + rest_len
                var merged = unsafe_alloc[UInt8](merged_len + 1)
                for mi in range(inc_len):
                    merged[unsafe_offset=mi] = sub_handle[unsafe_offset=0].buffer[unsafe_offset=mi]
                for mi in range(rest_len):
                    merged[unsafe_offset=inc_len + mi] = handle[unsafe_offset=0].buffer[unsafe_offset=rest_start + mi]
                merged[unsafe_offset=merged_len] = UInt8(0)
                if _is_real_ptr(handle[unsafe_offset=0].buffer):
                    handle[unsafe_offset=0].buffer.unsafe_free()
                handle[unsafe_offset=0].buffer = merged
                handle[unsafe_offset=0].total_bytes = Int32(merged_len)
                handle[unsafe_offset=0].cursor = Int32(0)
                handle[unsafe_offset=0].is_at_end = Int32(0)
            else:
                var inc_str = String(unsafe_from_utf8_ptr=inc_name.as_imm())
                if inc_str.endswith(".xz"):
                    print("Warning: cannot open include (decompress first with xz -dk):", inc_str)
                elif inc_str.endswith(".gz"):
                    print("Warning: cannot open include (gzip decompression failed):", inc_str)
                else:
                    print("Warning: cannot open include:", inc_str)
            scanner_free(sub_handle)
            if ends_gz:
                stripped.unsafe_free()
            inc_name.unsafe_free(); inc_path.unsafe_free()
        elif _psc_streq(kw_buf, "Material"):
            _psc_handle_make_named_material(handle, s, True)
            s[unsafe_offset=0].cur_attr.mat_idx = Int32(len(s[unsafe_offset=0].named_materials)) - Int32(1)
        elif _psc_streq(kw_buf, "MakeNamedMedium"):
            handle_named_medium(handle, s)
        elif _psc_streq(kw_buf, "MediumInterface"):
            handle_medium_interface(handle, s)
        elif _psc_streq(kw_buf, "ConcatTransform"):
            _ = scanner_scan_char(handle, UInt8(91))  # '['
            var tmp = unsafe_alloc[Float32](16)
            for i in range(16):
                _ = scanner_scan_float(handle, tmp.unsafe_offset(i))
            _ = scanner_scan_char(handle, UInt8(93))  # ']'
            var result = unsafe_alloc[Float32](16)
            matrix_multiply(s[unsafe_offset=0].ctm.unsafe_ptr(), tmp, result)
            for i in range(16):
                s[unsafe_offset=0].ctm[i] = result[unsafe_offset=i]
            tmp.unsafe_free(); result.unsafe_free()
        else:
            _ = scanner_parse_quoted_string(handle, kw_buf, 256)
            _psc_skip_params(handle)

    kw_buf.unsafe_free()
    ws_delims.unsafe_free()

# ── Camera/film matrix helpers ────────────────────────────────────────────────

def make_perspective_matrix(fov_deg: Float32, near: Float32,
                         dst: Pointer[Float32, MutUntrackedOrigin]):
    var half_rad = fov_deg * PI / Float32(360)
    var inv_tan = Float32(1) / tan(half_rad)
    var far = fov_deg
    var t22 = far / (far - near)
    var t23 = -(far * near) / (far - near)
    for i in range(16):
        dst[unsafe_offset=i] = Float32(0)
    dst[unsafe_offset=0]  = inv_tan
    dst[unsafe_offset=5]  = inv_tan
    dst[unsafe_offset=10] = t22
    dst[unsafe_offset=11] = Float32(1)
    dst[unsafe_offset=14] = t23

def make_screen_to_raster(fw: Int32, fh: Int32,
                               smin_x: Float32, smax_x: Float32,
                               smin_y: Float32, smax_y: Float32,
                               dst: Pointer[Float32, MutUntrackedOrigin]):
    var sx = Float32(fw) / (smax_x - smin_x)
    var sy = Float32(fh) / (smin_y - smax_y)
    var tx = -smin_x * sx
    var ty = -smax_y * sy
    for i in range(16):
        dst[unsafe_offset=i] = Float32(0)
    dst[unsafe_offset=0]  = sx
    dst[unsafe_offset=5]  = sy
    dst[unsafe_offset=10] = Float32(1)
    dst[unsafe_offset=15] = Float32(1)
    dst[unsafe_offset=12] = tx
    dst[unsafe_offset=13] = ty

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
    out_first: Pointer[Int32, MutUntrackedOrigin],
    out_count: Pointer[Int32, MutUntrackedOrigin],
    write: Bool,
) -> Int:
    """Greedy-merge adjacent pieces of one curve into flat runs. Returns the
    number of groups. If write=True, fills out_first/out_count (each must
    have capacity >= n_pieces) with (first_piece, piece_count) per group; if
    write=False the out pointers are ignored — used for a cheap first pass
    to size the final arrays before allocating them."""
    var pts = Array[Vec3f, CURVE_N_PIECES + 1](fill=Vec3f(0, 0, 0))
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
            out_first[unsafe_offset=n_groups] = Int32(start)
            out_count[unsafe_offset=n_groups] = Int32(end - start)
        n_groups += 1
        start = end
    return n_groups

# ── Film sensor: white balance ────────────────────────────────────────────────
# pbrt's PixelSensor applies a von Kries (Bradford) chromatic adaptation from
# the illuminant named by the film's "whitebalance" colour temperature to the
# output colour space's white (D65 for sRGB), then scales by an imaging ratio
# of exposuretime * iso / 100. Both are reproduced exactly here for pbrt's
# DEFAULT sensor; a named sensor's measured response curves are not ported
# (see the warning in the Film handler). Returns a 3x3 row-major RGB->RGB
# matrix in lanes 0..8 -- identity when whitebalance is 0 (pbrt's default,
# meaning "no white balancing").

def _film_white_balance_matrix(temp_k: Float32) -> SIMD[DType.float32, 16]:
    var m = SIMD[DType.float32, 16](0)
    m[0] = Float32(1); m[4] = Float32(1); m[8] = Float32(1)
    if temp_k <= Float32(0):
        return m
    # CIE D-illuminant chromaticity locus for the requested temperature.
    var t = Float64(temp_k)
    var x: Float64
    if t <= 7000.0:
        x = -4.6070e9/(t*t*t) + 2.9678e6/(t*t) + 0.09911e3/t + 0.244063
    else:
        x = -2.0064e9/(t*t*t) + 1.9018e6/(t*t) + 0.24748e3/t + 0.237040
    var y = -3.000*x*x + 2.870*x - 0.275

    var src = Array[Float64, 3](fill=0.0)
    src[0] = x/y; src[1] = 1.0; src[2] = (1.0 - x - y)/y
    var dst = Array[Float64, 3](fill=0.0)
    var dx = 0.3127; var dy = 0.3290          # sRGB white (D65)
    dst[0] = dx/dy; dst[1] = 1.0; dst[2] = (1.0 - dx - dy)/dy

    # Bradford LMS<->XYZ, the same matrices pbrt uses.
    var L = Array[Float64, 9](fill=0.0)
    L[0]= 0.8951; L[1]= 0.2664; L[2]=-0.1614
    L[3]=-0.7502; L[4]= 1.7135; L[5]= 0.0367
    L[6]= 0.0389; L[7]=-0.0685; L[8]= 1.0296
    var Li = Array[Float64, 9](fill=0.0)
    Li[0]= 0.986993;   Li[1]=-0.147054;  Li[2]= 0.159963
    Li[3]= 0.432305;   Li[4]= 0.51836;   Li[5]= 0.0492912
    Li[6]=-0.00852866; Li[7]= 0.0400428; Li[8]= 0.968487
    # sRGB primaries.
    var XR = Array[Float64, 9](fill=0.0)
    XR[0]=0.4124564; XR[1]=0.3575761; XR[2]=0.1804375
    XR[3]=0.2126729; XR[4]=0.7151522; XR[5]=0.0721750
    XR[6]=0.0193339; XR[7]=0.1191920; XR[8]=0.9503041
    var RX = Array[Float64, 9](fill=0.0)
    RX[0]= 3.2404542; RX[1]=-1.5371385; RX[2]=-0.4985314
    RX[3]=-0.9692660; RX[4]= 1.8760108; RX[5]= 0.0415560
    RX[6]= 0.0556434; RX[7]=-0.2040259; RX[8]= 1.0572252

    var sl = Array[Float64, 3](fill=0.0)
    var dl = Array[Float64, 3](fill=0.0)
    for r in range(3):
        sl[r] = L[r*3]*src[0] + L[r*3+1]*src[1] + L[r*3+2]*src[2]
        dl[r] = L[r*3]*dst[0] + L[r*3+1]*dst[1] + L[r*3+2]*dst[2]

    # M = RGBfromXYZ * (XYZfromLMS * diag(dl/sl) * LMSfromXYZ) * XYZfromRGB
    var A = Array[Float64, 9](fill=0.0)     # XYZfromLMS * diag
    for r in range(3):
        for c in range(3):
            var g = dl[c] / sl[c] if sl[c] != 0.0 else 1.0
            A[r*3+c] = Li[r*3+c] * g
    var B = Array[Float64, 9](fill=0.0)     # A * LMSfromXYZ
    for r in range(3):
        for c in range(3):
            var acc = 0.0
            for k in range(3): acc += A[r*3+k] * L[k*3+c]
            B[r*3+c] = acc
    var C = Array[Float64, 9](fill=0.0)     # B * XYZfromRGB
    for r in range(3):
        for c in range(3):
            var acc = 0.0
            for k in range(3): acc += B[r*3+k] * XR[k*3+c]
            C[r*3+c] = acc
    for r in range(3):
        for c in range(3):
            var acc = 0.0
            for k in range(3): acc += RX[r*3+k] * C[k*3+c]
            m[r*3+c] = Float32(acc)
    return m

# ── Scene finalization ────────────────────────────────────────────────────────

def finalize_scene(s: Pointer[SceneParseState, MutUntrackedOrigin],
                 psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
                 verbose: Bool = False):

    # ---- Camera matrices ----
    var c2w = unsafe_alloc[Float32](16)
    var cam2w_tmp = unsafe_alloc[Float32](16)
    for i in range(16): cam2w_tmp[unsafe_offset=i] = s[unsafe_offset=0].cam2w_raw[i]
    _ = matrix_invert(cam2w_tmp, c2w)
    cam2w_tmp.unsafe_free()
    psc[unsafe_offset=0].camera_to_world = c2w

    if verbose:
        print("=== GONZALES DEBUG: Scene Summary ===")
        print("  Camera position (c2w col3):", c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])
        print("  Camera forward (-Z):", -c2w[unsafe_offset=8], -c2w[unsafe_offset=9], -c2w[unsafe_offset=10])
        print("  Camera FOV:", s[unsafe_offset=0].camera_fov)
        print("  Film:", s[unsafe_offset=0].film_w, "x", s[unsafe_offset=0].film_h)
        print("  Meshes:", len(s[unsafe_offset=0].meshes))
        print("  Named materials:", len(s[unsafe_offset=0].named_materials))
        print("=== END DEBUG ===")

    var cts = unsafe_alloc[Float32](16)
    make_perspective_matrix(s[unsafe_offset=0].camera_fov, Float32(0.01), cts)

    var frame = Float32(s[unsafe_offset=0].film_w) / Float32(s[unsafe_offset=0].film_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = unsafe_alloc[Float32](16)
    make_screen_to_raster(s[unsafe_offset=0].film_w, s[unsafe_offset=0].film_h,
                                smin_x, smax_x, smin_y, smax_y, str_mat)

    var rts = unsafe_alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)

    var cts_inv = unsafe_alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)

    var r2c = unsafe_alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c)
    psc[unsafe_offset=0].raster_to_camera = r2c

    cts.unsafe_free(); str_mat.unsafe_free(); rts.unsafe_free(); cts_inv.unsafe_free()

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
    for i in range(len(s[unsafe_offset=0].meshes)):
        if s[unsafe_offset=0].meshes[i].mat_idx == Int32(-1) and not s[unsafe_offset=0].meshes[i].is_area_light:
            needs_default_mat = True
            break
    if not needs_default_mat:
        for i in range(len(s[unsafe_offset=0].spheres_mat)):
            if s[unsafe_offset=0].spheres_mat[i] == Int32(-1):
                needs_default_mat = True
                break
    if not needs_default_mat:
        for i in range(len(s[unsafe_offset=0].curves_mat)):
            var curve_is_al = i < len(s[unsafe_offset=0].curves_al) and s[unsafe_offset=0].curves_al[i]
            if s[unsafe_offset=0].curves_mat[i] == Int32(-1) and not curve_is_al:
                needs_default_mat = True
                break
    var default_mat_idx = Int32(-1)
    if needs_default_mat:
        var default_nm = NamedMaterial(String("__default_diffuse__"))
        default_nm.kind = MatKind.diffuse
        default_nm.albedo = RGB(Float32(0.5), Float32(0.5), Float32(0.5))
        s[unsafe_offset=0].named_materials.append(default_nm^)
        default_mat_idx = Int32(len(s[unsafe_offset=0].named_materials) - 1)

    var n_regular = len(s[unsafe_offset=0].named_materials)

    var n_al_mesh = 0
    for i in range(len(s[unsafe_offset=0].meshes)):
        if s[unsafe_offset=0].meshes[i].is_area_light:
            n_al_mesh += 1

    # Curve area lights (see curves_al/curves_al_rgb) get their own synthetic
    # material slots too, right after the mesh area-light slots — one per
    # emissive curve *segment* (a `Shape "curve"` directive splits into
    # several local B-spline segments, see curves_mat; they all share the
    # same cur_attr.al_rgb at parse time, so this is simply the simplest
    # correct granularity, not a meaningful dedup opportunity lost).
    var n_al_curve = 0
    for i in range(len(s[unsafe_offset=0].curves_al)):
        if s[unsafe_offset=0].curves_al[i]:
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
        var mpath = s[unsafe_offset=0].named_materials[i].measured_bsdf_path
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
    psc[unsafe_offset=0].measured_count = Int32(len(measured_list))
    var measured_brdfs_buf = unsafe_alloc[MeasuredBRDF_C](max(len(measured_list), 1))
    for i in range(len(measured_list)):
        measured_brdfs_buf[unsafe_offset=i] = measured_list[i]
    psc[unsafe_offset=0].measured_brdfs = measured_brdfs_buf

    var mats = unsafe_alloc[Material_C](max(n_mats, 1))
    for i in range(n_regular):
        var nm3 = s[unsafe_offset=0].named_materials[i]
        var material_kind = nm3.kind
        var ior = nm3.ior
        mats[unsafe_offset=i].type = material_kind
        mats[unsafe_offset=i].tex_idx = nm3.tex_idx
        mats[unsafe_offset=i].roughU  = nm3.roughness_u
        mats[unsafe_offset=i].roughV  = nm3.roughness_v
        mats[unsafe_offset=i].normal_tex_idx = nm3.normal_tex_idx
        mats[unsafe_offset=i].bump_tex_idx = nm3.bump_tex_idx
        mats[unsafe_offset=i].bump_scale = nm3.bump_scale
        mats[unsafe_offset=i].tex_scale = nm3.tex_scale
        mats[unsafe_offset=i].sss_mean_refl = nm3.sss_mean_refl
        mats[unsafe_offset=i].tex_bias = nm3.tex_bias
        mats[unsafe_offset=i].rough_tex_idx = nm3.rough_tex_idx
        mats[unsafe_offset=i].medium_interface_idx = Int32(-1)
        # `mats` comes from unsafe_alloc, which does NOT zero. Every other
        # field is assigned below, but sss_boundary was only ever written on
        # the medium-interface duplication path further down -- so a material
        # with no medium kept whatever byte happened to be on the heap, and
        # that byte differs per process run. It is read as a plain
        # `!= 0` flag, so ~2/3 of this corpus's materials claimed to be
        # subsurface boundaries at random. See
        # project_sppm_nondeterministic_photon_pass memory: barcelona's water
        # flipped between refracting and absorbing every photon on it.
        mats[unsafe_offset=i].sss_boundary = Int8(0)
        mats[unsafe_offset=i]._pad1 = Int8(0)
        mats[unsafe_offset=i]._pad2 = Int8(0)
        if nm3.measured_bsdf_path == "":
            mats[unsafe_offset=i].measured_idx = Int32(-1)
        else:
            var midx = Int32(-1)
            for j in range(len(measured_paths)):
                if measured_paths[j] == nm3.measured_bsdf_path:
                    if measured_ok[j]:
                        midx = Int32(j)
                    break
            mats[unsafe_offset=i].measured_idx = midx
            if midx >= Int32(0):
                # Real MeasuredBxDF loaded successfully -- route through the
                # real algorithm (shading.mojo's shade_measured) instead of
                # material_builder.mojo's achromatic rough-conductor
                # approximation (material_kind, still MatKind.conductor,
                # stays as the graceful fallback for a load failure above).
                mats[unsafe_offset=i].type = MatKind.measured
        mats[unsafe_offset=i].checker_tex1   = nm3.checker_tex1
        mats[unsafe_offset=i].checker_tex2   = nm3.checker_tex2
        mats[unsafe_offset=i].checker_uscale = nm3.checker_uscale
        mats[unsafe_offset=i].checker_vscale = nm3.checker_vscale
        if material_kind == MatKind.dielectric:
            mats[unsafe_offset=i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[unsafe_offset=i].emission = RGB(Float32(0))
        elif material_kind == MatKind.coated_diffuse:
            mats[unsafe_offset=i].albedo = nm3.albedo
            mats[unsafe_offset=i].emission = RGB(ior, Float32(0), Float32(0))
        elif material_kind == MatKind.coated_conductor:
            mats[unsafe_offset=i].albedo = nm3.albedo
            mats[unsafe_offset=i].emission = RGB(ior, Float32(0), Float32(0))
        elif material_kind == MatKind.thin_dielectric:
            mats[unsafe_offset=i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[unsafe_offset=i].emission = RGB(Float32(0))
        elif material_kind == MatKind.mix:
            var idx1 = Int32(0)
            var idx2 = Int32(0)
            for j in range(n_regular):
                if s[unsafe_offset=0].named_materials[j].name == nm3.mix_name1: idx1 = Int32(j)
                if s[unsafe_offset=0].named_materials[j].name == nm3.mix_name2: idx2 = Int32(j)
            mats[unsafe_offset=i].tex_idx = (idx2 << 16) | (idx1 & Int32(0xFFFF))
            mats[unsafe_offset=i].roughU  = nm3.mix_amount
            mats[unsafe_offset=i].albedo  = nm3.albedo
            mats[unsafe_offset=i].emission = RGB(Float32(0))
        elif material_kind == MatKind.diffuse_transmit:
            mats[unsafe_offset=i].albedo = nm3.albedo
            mats[unsafe_offset=i].emission = nm3.transmittance
        elif material_kind == MatKind.hair:
            mats[unsafe_offset=i].albedo = nm3.albedo   # sigma_a per channel (absorption coefficient)
            mats[unsafe_offset=i].emission = RGB(Float32(1.55), Float32(0), Float32(0))  # IOR for hair cuticle
            if nm3.roughness_u == Float32(0):
                mats[unsafe_offset=i].roughU = Float32(0.3)  # betaM default
            else:
                mats[unsafe_offset=i].roughU = nm3.roughness_u
            if nm3.roughness_v == Float32(0):
                mats[unsafe_offset=i].roughV = Float32(0.3)  # betaN default
            else:
                mats[unsafe_offset=i].roughV = nm3.roughness_v
            mats[unsafe_offset=i].type = MatKind.hair
        else:
            mats[unsafe_offset=i].albedo = nm3.albedo
            mats[unsafe_offset=i].emission = RGB(Float32(0))

    # ---- Meshes + area lights ----
    var n_meshes = len(s[unsafe_offset=0].meshes)
    var meshes   = unsafe_alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts  = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](max(n_meshes, 1))
    var out_vis  = unsafe_alloc[Pointer[Int64, MutUntrackedOrigin]](max(n_meshes, 1))
    var out_fis  = unsafe_alloc[Pointer[Int64, MutUntrackedOrigin]](max(n_meshes, 1))
    var out_nv    = unsafe_alloc[Int32](max(n_meshes, 1))
    var out_nt    = unsafe_alloc[Int32](max(n_meshes, 1))
    var out_uv_nv = unsafe_alloc[Int32](max(n_meshes, 1))
    var out_nrm_nv = unsafe_alloc[Int32](max(n_meshes, 1))

    # al_list (used for NEE light sampling) covers mesh area lights (kind=0,
    # filled below) AND curve area lights (kind=1, appended once curve_buf
    # is built further down in "Native curves") — sized for both up front
    # since it's one contiguous allocation.
    var al_list  = unsafe_alloc[AreaLight_C](max(n_al_mesh + n_al_curve, 1))
    var al_count = Int32(0)
    var al_mat_base = n_regular

    for i in range(n_meshes):
        ref ma = s[unsafe_offset=0].meshes[i]
        var nv = len(ma.points) // 4
        var nt = len(ma.face_idxs)
        var pts_c = unsafe_alloc[Float32](nv * 4)
        for vi in range(nv * 4): pts_c[unsafe_offset=vi] = ma.points[vi]
        var vis_c = unsafe_alloc[Int64](nt * 3)
        for ti2 in range(nt * 3): vis_c[unsafe_offset=ti2] = ma.vert_idxs[ti2]
        var fis_c = unsafe_alloc[Int64](nt)
        for ti2 in range(nt): fis_c[unsafe_offset=ti2] = ma.face_idxs[ti2]
        out_pts[unsafe_offset=i] = pts_c
        out_vis[unsafe_offset=i] = vis_c
        out_fis[unsafe_offset=i] = fis_c
        out_nv[unsafe_offset=i]  = Int32(nv)
        out_nt[unsafe_offset=i]  = Int32(nt)
        meshes[unsafe_offset=i].points        = pts_c
        meshes[unsafe_offset=i].vertexIndices = vis_c
        meshes[unsafe_offset=i].faceIndices   = fis_c
        if len(ma.uvs) >= nv * 2:
            var uv_c = unsafe_alloc[Float32](nv * 2)
            for ui in range(nv * 2): uv_c[unsafe_offset=ui] = ma.uvs[ui]
            meshes[unsafe_offset=i].uvs = uv_c
            out_uv_nv[unsafe_offset=i] = Int32(nv)
        else:
            meshes[unsafe_offset=i].uvs = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            out_uv_nv[unsafe_offset=i] = Int32(0)
        if len(ma.normals) >= nv * 3:
            var nrm_c = unsafe_alloc[Float32](nv * 3)
            for ni in range(nv * 3): nrm_c[unsafe_offset=ni] = ma.normals[ni]
            meshes[unsafe_offset=i].normals = nrm_c
            out_nrm_nv[unsafe_offset=i] = Int32(nv)
        else:
            meshes[unsafe_offset=i].normals = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            out_nrm_nv[unsafe_offset=i] = Int32(0)

        if ma.is_area_light:
            var al_idx = Int(al_count)
            var em = ma.al_rgb
            var t_area  = Float32(0.0)
            for ti in range(nt):
                var vi0 = Int(vis_c[unsafe_offset=ti*3+0]) * 4
                var vi1 = Int(vis_c[unsafe_offset=ti*3+1]) * 4
                var vi2 = Int(vis_c[unsafe_offset=ti*3+2]) * 4
                var ex = pts_c[unsafe_offset=vi1+0] - pts_c[unsafe_offset=vi0+0]
                var ey = pts_c[unsafe_offset=vi1+1] - pts_c[unsafe_offset=vi0+1]
                var ez = pts_c[unsafe_offset=vi1+2] - pts_c[unsafe_offset=vi0+2]
                var fx = pts_c[unsafe_offset=vi2+0] - pts_c[unsafe_offset=vi0+0]
                var fy = pts_c[unsafe_offset=vi2+1] - pts_c[unsafe_offset=vi0+1]
                var fz = pts_c[unsafe_offset=vi2+2] - pts_c[unsafe_offset=vi0+2]
                var cxv = ey*fz - ez*fy
                var cyv = ez*fx - ex*fz
                var czv = ex*fy - ey*fx
                t_area += Float32(0.5) * sqrt(cxv*cxv + cyv*cyv + czv*czv)
            al_list[unsafe_offset=al_idx].meshIdx    = Int32(i)
            al_list[unsafe_offset=al_idx].n_tris     = Int32(nt)
            al_list[unsafe_offset=al_idx].emission   = em
            al_list[unsafe_offset=al_idx].total_area = t_area
            al_list[unsafe_offset=al_idx].kind       = Int8(0)
            mats[unsafe_offset=al_mat_base + al_idx].type     = Int8(2)
            mats[unsafe_offset=al_mat_base + al_idx].albedo   = RGB(Float32(0))
            mats[unsafe_offset=al_mat_base + al_idx].emission = em
            mats[unsafe_offset=al_mat_base + al_idx].tex_idx  = Int32(-1)
            mats[unsafe_offset=al_mat_base + al_idx].roughU   = Float32(0)
            mats[unsafe_offset=al_mat_base + al_idx].roughV   = Float32(0)
            mats[unsafe_offset=al_mat_base + al_idx].normal_tex_idx = Int32(-1)
            mats[unsafe_offset=al_mat_base + al_idx].bump_tex_idx = Int32(-1)
            mats[unsafe_offset=al_mat_base + al_idx].bump_scale = Float32(1)
            mats[unsafe_offset=al_mat_base + al_idx].tex_scale = RGB(Float32(1))
            mats[unsafe_offset=al_mat_base + al_idx].sss_mean_refl = RGB(Float32(1))
            mats[unsafe_offset=al_mat_base + al_idx].tex_bias = RGB(Float32(0))
            mats[unsafe_offset=al_mat_base + al_idx].rough_tex_idx = Int32(-1)
            mats[unsafe_offset=al_mat_base + al_idx].medium_interface_idx = Int32(-1)
            mats[unsafe_offset=al_mat_base + al_idx].measured_idx = Int32(-1)
            al_count += 1

    # ---- Curve area light material slots ----
    # Each emissive curve segment gets its own synthetic material slot here
    # (used by the direct-hit path in shade_nee_core), same as before. Their
    # AreaLight_C/NEE entries (al_list[n_al_mesh:]) are appended further down
    # in "Native curves" once curve_buf/curve_n_pieces exist — total_area
    # needs the curve's actual piece tessellation (curve_light_tube_area).
    var curve_al_mat_idx = unsafe_alloc[Int32](max(len(s[unsafe_offset=0].curves_al), 1))
    var curve_al_running = Int32(0)
    for ci in range(len(s[unsafe_offset=0].curves_al)):
        if s[unsafe_offset=0].curves_al[ci]:
            var slot = al_mat_base + n_al_mesh + Int(curve_al_running)
            curve_al_mat_idx[unsafe_offset=ci] = Int32(slot)
            var em = s[unsafe_offset=0].curves_al_rgb[ci]
            mats[unsafe_offset=slot].type     = MatKind.area_light
            mats[unsafe_offset=slot].albedo   = RGB(Float32(0))
            mats[unsafe_offset=slot].emission = em
            mats[unsafe_offset=slot].tex_idx  = Int32(-1)
            mats[unsafe_offset=slot].roughU   = Float32(0)
            mats[unsafe_offset=slot].roughV   = Float32(0)
            mats[unsafe_offset=slot].normal_tex_idx = Int32(-1)
            mats[unsafe_offset=slot].bump_tex_idx = Int32(-1)
            mats[unsafe_offset=slot].bump_scale = Float32(1)
            mats[unsafe_offset=slot].tex_scale = RGB(Float32(1))
            mats[unsafe_offset=slot].tex_bias = RGB(Float32(0))
            mats[unsafe_offset=slot].rough_tex_idx = Int32(-1)
            mats[unsafe_offset=slot].medium_interface_idx = Int32(-1)
            mats[unsafe_offset=slot].measured_idx = Int32(-1)
            curve_al_running += 1
        else:
            curve_al_mat_idx[unsafe_offset=ci] = Int32(-1)

    # ---- BVH construction ----
    var n_with_mi = 0
    for mi in range(n_meshes):
        if s[unsafe_offset=0].meshes[mi].inside_medium >= Int32(0) or s[unsafe_offset=0].meshes[mi].outside_medium >= Int32(0):
            n_with_mi += 1
    for si in range(len(s[unsafe_offset=0].spheres_cx)):
        if s[unsafe_offset=0].spheres_inside_med[si] >= Int32(0) or s[unsafe_offset=0].spheres_outside_med[si] >= Int32(0):
            n_with_mi += 1

    if n_with_mi > 0:
        var expanded_n = n_mats + n_with_mi
        var new_mats = unsafe_alloc[Material_C](expanded_n)
        for ci in range(n_mats):
            new_mats[unsafe_offset=ci] = mats[unsafe_offset=ci]
        mats.unsafe_free()
        mats = new_mats

        var iface_buf = unsafe_alloc[MediumInterface_C](n_with_mi)
        var dup_idx = n_mats
        var iface_idx = 0

        # An interface whose INTERIOR is a subsurface medium marks its
        # material, so shade_dielectric can keep the resulting boundary
        # events off the maxdepth budget (see Material_C.sss_boundary).
        def _ins_is_sss(ins: Int32) {imm} -> Int8:
            if ins < Int32(0): return Int8(0)
            if Int(ins) >= len(s[unsafe_offset=0].med_is_sss): return Int8(0)
            return Int8(1) if s[unsafe_offset=0].med_is_sss[Int(ins)] != Int32(0) else Int8(0)

        for mi in range(n_meshes):
            var ins = s[unsafe_offset=0].meshes[mi].inside_medium
            var out = s[unsafe_offset=0].meshes[mi].outside_medium
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[unsafe_offset=0].meshes[mi].mat_idx)
            if orig_mat < 0: continue
            mats[unsafe_offset=dup_idx] = mats[unsafe_offset=orig_mat]
            mats[unsafe_offset=dup_idx].medium_interface_idx = Int32(iface_idx)
            mats[unsafe_offset=dup_idx].sss_boundary = _ins_is_sss(ins)
            iface_buf[unsafe_offset=iface_idx] = MediumInterface_C(ins, out)
            s[unsafe_offset=0].meshes[mi].mat_idx = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        for si in range(len(s[unsafe_offset=0].spheres_cx)):
            var ins = s[unsafe_offset=0].spheres_inside_med[si]
            var out = s[unsafe_offset=0].spheres_outside_med[si]
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[unsafe_offset=0].spheres_mat[si])
            if orig_mat < 0: continue
            mats[unsafe_offset=dup_idx] = mats[unsafe_offset=orig_mat]
            mats[unsafe_offset=dup_idx].medium_interface_idx = Int32(iface_idx)
            mats[unsafe_offset=dup_idx].sss_boundary = _ins_is_sss(ins)
            iface_buf[unsafe_offset=iface_idx] = MediumInterface_C(ins, out)
            s[unsafe_offset=0].spheres_mat[si] = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        n_mats = dup_idx
        psc[unsafe_offset=0].medium_ifaces = iface_buf
        psc[unsafe_offset=0].medium_iface_count = Int32(iface_idx)
    else:
        psc[unsafe_offset=0].medium_ifaces = Pointer[MediumInterface_C, MutUntrackedOrigin].unsafe_dangling()
        psc[unsafe_offset=0].medium_iface_count = Int32(0)

    var total_tris = Int32(0)
    for i in range(n_meshes):
        if not s[unsafe_offset=0].meshes[i].is_object_template:
            total_tris += Int32(len(s[unsafe_offset=0].meshes[i].face_idxs))

    # Native curves: precompute per-curve piece count via the same flatness
    # test used for the Curve_C upload below, then greedily merge adjacent
    # flat-enough pieces of each curly curve into single BVH leaves (see
    # _curve_greedy_groups) so the leaf count stays close to the number of
    # visually-distinct bends, not always CURVE_N_PIECES.
    var n_curves = Int32(len(s[unsafe_offset=0].curves_mat))
    var curve_n_pieces = unsafe_alloc[Int32](max(Int(n_curves), 1))
    var curve_group_base = unsafe_alloc[Int32](max(Int(n_curves), 1))
    var total_curve_groups = Int32(0)
    # Never dereferenced (the counting pass below only counts groups; the
    # write=False branch of _curve_greedy_groups skips all writes) — just
    # needs to be a valid, non-dangling pointer to satisfy the signature.
    var count_pass_scratch = unsafe_alloc[Int32](1)
    for i in range(Int(n_curves)):
        var cb = i * 12
        var cx0 = s[unsafe_offset=0].curves_cp[cb+0]; var cy0 = s[unsafe_offset=0].curves_cp[cb+1]; var cz0 = s[unsafe_offset=0].curves_cp[cb+2]
        var cx3 = s[unsafe_offset=0].curves_cp[cb+9]; var cy3 = s[unsafe_offset=0].curves_cp[cb+10]; var cz3 = s[unsafe_offset=0].curves_cp[cb+11]
        var chord_x = cx3 - cx0; var chord_y = cy3 - cy0; var chord_z = cz3 - cz0
        var chord_len = sqrt(chord_x*chord_x + chord_y*chord_y + chord_z*chord_z)
        var max_dev = Float32(0.0)
        if chord_len > Float32(1e-8):
            var dcx = chord_x / chord_len; var dcy = chord_y / chord_len; var dcz = chord_z / chord_len
            for k in range(1, 3):
                var vx = s[unsafe_offset=0].curves_cp[cb+k*3+0] - cx0
                var vy = s[unsafe_offset=0].curves_cp[cb+k*3+1] - cy0
                var vz = s[unsafe_offset=0].curves_cp[cb+k*3+2] - cz0
                var ccx = vy*dcz - vz*dcy
                var ccy = vz*dcx - vx*dcz
                var ccz = vx*dcy - vy*dcx
                var dist = sqrt(ccx*ccx + ccy*ccy + ccz*ccz)
                if dist > max_dev: max_dev = dist
        var maxw = max(s[unsafe_offset=0].curves_w0[i], s[unsafe_offset=0].curves_w1[i])
        curve_n_pieces[unsafe_offset=i] = Int32(1) if max_dev < maxw * Float32(0.5) else Int32(CURVE_N_PIECES)

        var curve_i = Curve_C(
            Point3f(s[unsafe_offset=0].curves_cp[cb+0], s[unsafe_offset=0].curves_cp[cb+1], s[unsafe_offset=0].curves_cp[cb+2]),
            Point3f(s[unsafe_offset=0].curves_cp[cb+3], s[unsafe_offset=0].curves_cp[cb+4], s[unsafe_offset=0].curves_cp[cb+5]),
            Point3f(s[unsafe_offset=0].curves_cp[cb+6], s[unsafe_offset=0].curves_cp[cb+7], s[unsafe_offset=0].curves_cp[cb+8]),
            Point3f(s[unsafe_offset=0].curves_cp[cb+9], s[unsafe_offset=0].curves_cp[cb+10], s[unsafe_offset=0].curves_cp[cb+11]),
            s[unsafe_offset=0].curves_w0[i], s[unsafe_offset=0].curves_w1[i], s[unsafe_offset=0].curves_mat[i], curve_n_pieces[unsafe_offset=i])
        var ngroups = _curve_greedy_groups(curve_i, Int(curve_n_pieces[unsafe_offset=i]), count_pass_scratch, count_pass_scratch, False)
        curve_group_base[unsafe_offset=i] = total_curve_groups
        total_curve_groups += Int32(ngroups)

    var total_instances = Int32(len(s[unsafe_offset=0].instance_template_idx))
    var total_prims_gpu = total_tris + total_curve_groups
    var total_prims = total_prims_gpu + total_instances

    var prim_bounds = unsafe_alloc[Float32](Int(total_prims) * 6)
    var tri_mesh    = unsafe_alloc[Int32](max(Int(total_tris), 1))
    var tri_local   = unsafe_alloc[Int32](max(Int(total_tris), 1))

    var flat_idx = Int32(0)
    for mi in range(n_meshes):
        if s[unsafe_offset=0].meshes[mi].is_object_template:
            continue
        var pts = out_pts[unsafe_offset=mi]
        var vis = out_vis[unsafe_offset=mi]
        var nt  = Int(out_nt[unsafe_offset=mi])
        for ti in range(nt):
            var v0 = Int(vis[unsafe_offset=ti*3+0]) * 4
            var v1 = Int(vis[unsafe_offset=ti*3+1]) * 4
            var v2 = Int(vis[unsafe_offset=ti*3+2]) * 4
            var x0 = pts[unsafe_offset=v0]; var y0 = pts[unsafe_offset=v0+1]; var z0 = pts[unsafe_offset=v0+2]
            var x1 = pts[unsafe_offset=v1]; var y1 = pts[unsafe_offset=v1+1]; var z1 = pts[unsafe_offset=v1+2]
            var x2 = pts[unsafe_offset=v2]; var y2 = pts[unsafe_offset=v2+1]; var z2 = pts[unsafe_offset=v2+2]
            var b = Int(flat_idx) * 6
            prim_bounds[unsafe_offset=b+0] = min(x0, min(x1, x2))
            prim_bounds[unsafe_offset=b+1] = min(y0, min(y1, y2))
            prim_bounds[unsafe_offset=b+2] = min(z0, min(z1, z2))
            prim_bounds[unsafe_offset=b+3] = max(x0, max(x1, x2))
            prim_bounds[unsafe_offset=b+4] = max(y0, max(y1, y2))
            prim_bounds[unsafe_offset=b+5] = max(z0, max(z1, z2))
            tri_mesh[unsafe_offset=Int(flat_idx)]  = Int32(mi)
            tri_local[unsafe_offset=Int(flat_idx)] = Int32(ti)
            flat_idx += 1

    # Native curve groups: one BVH leaf per greedily-merged run of pieces
    # (see _curve_greedy_groups), each with a tight AABB — the union of that
    # run's individual piece bounds, still far tighter than the old
    # whole-segment hull since a run rarely spans the entire curly curve.
    var group_curve_idx = unsafe_alloc[Int32](max(Int(total_curve_groups), 1))
    var group_id2 = unsafe_alloc[Int32](max(Int(total_curve_groups), 1))  # packed first_piece*8 + piece_count
    var group_first_scratch = unsafe_alloc[Int32](CURVE_N_PIECES)
    var group_count_scratch = unsafe_alloc[Int32](CURVE_N_PIECES)
    for ci in range(Int(n_curves)):
        var base = ci * 12
        var curve_i = Curve_C(
            Point3f(s[unsafe_offset=0].curves_cp[base+0], s[unsafe_offset=0].curves_cp[base+1], s[unsafe_offset=0].curves_cp[base+2]),
            Point3f(s[unsafe_offset=0].curves_cp[base+3], s[unsafe_offset=0].curves_cp[base+4], s[unsafe_offset=0].curves_cp[base+5]),
            Point3f(s[unsafe_offset=0].curves_cp[base+6], s[unsafe_offset=0].curves_cp[base+7], s[unsafe_offset=0].curves_cp[base+8]),
            Point3f(s[unsafe_offset=0].curves_cp[base+9], s[unsafe_offset=0].curves_cp[base+10], s[unsafe_offset=0].curves_cp[base+11]),
            s[unsafe_offset=0].curves_w0[ci], s[unsafe_offset=0].curves_w1[ci], s[unsafe_offset=0].curves_mat[ci], curve_n_pieces[unsafe_offset=ci])
        var ngroups = _curve_greedy_groups(curve_i, Int(curve_n_pieces[unsafe_offset=ci]), group_first_scratch, group_count_scratch, True)
        for g in range(ngroups):
            var global_group = Int(curve_group_base[unsafe_offset=ci]) + g
            var first_piece = Int(group_first_scratch[unsafe_offset=g])
            var piece_count = Int(group_count_scratch[unsafe_offset=g])
            group_curve_idx[unsafe_offset=global_group] = Int32(ci)
            group_id2[unsafe_offset=global_group] = Int32(first_piece * 8 + piece_count)
            var (xmin, ymin, zmin, xmax, ymax, zmax) = curve_piece_bounds(curve_i, first_piece)
            for p in range(first_piece + 1, first_piece + piece_count):
                var (pxmin, pymin, pzmin, pxmax, pymax, pzmax) = curve_piece_bounds(curve_i, p)
                xmin = min(xmin, pxmin); ymin = min(ymin, pymin); zmin = min(zmin, pzmin)
                xmax = max(xmax, pxmax); ymax = max(ymax, pymax); zmax = max(zmax, pzmax)
            var b = (Int(total_tris) + global_group) * 6
            prim_bounds[unsafe_offset=b+0] = xmin; prim_bounds[unsafe_offset=b+1] = ymin; prim_bounds[unsafe_offset=b+2] = zmin
            prim_bounds[unsafe_offset=b+3] = xmax; prim_bounds[unsafe_offset=b+4] = ymax; prim_bounds[unsafe_offset=b+5] = zmax
    group_first_scratch.unsafe_free(); group_count_scratch.unsafe_free(); count_pass_scratch.unsafe_free()

    # ---- Object instancing: one BLAS per template, then a TLAS instance leaf
    # (transform + BLAS reference) per ObjectInstance placement — see
    # geometry.mojo's Instance_C docs. A BLAS is a private BVH2 built over
    # just that template's own mesh range, with ordinary type==0 PrimId_C
    # entries referencing the SAME GLOBAL `meshes` array (no per-BLAS mesh
    # storage, no geometry duplication).
    var n_templates = len(s[unsafe_offset=0].object_names)
    var blas_nodes_arr   = unsafe_alloc[Pointer[BVH2Node, MutUntrackedOrigin]](max(n_templates, 1))
    var blas_primids_arr = unsafe_alloc[Pointer[PrimId_C, MutUntrackedOrigin]](max(n_templates, 1))
    # Per-BLAS array lengths — the CPU traversal side never needs these (it
    # just walks from node/primid index 0, self-describing via each node's
    # offset/count), but GPU upload does: it copies each BLAS's arrays into
    # their own device buffers and needs to know how many bytes that is.
    var blas_node_counts   = unsafe_alloc[Int32](max(n_templates, 1))
    var blas_primid_counts = unsafe_alloc[Int32](max(n_templates, 1))
    var template_mesh_start = unsafe_alloc[Int32](max(n_templates, 1))
    var template_mesh_end   = unsafe_alloc[Int32](max(n_templates, 1))
    for tmpl in range(n_templates):
        var mstart = Int(s[unsafe_offset=0].object_mesh_start[tmpl])
        var mend   = Int(s[unsafe_offset=0].object_mesh_end[tmpl])
        template_mesh_start[unsafe_offset=tmpl] = Int32(mstart)
        template_mesh_end[unsafe_offset=tmpl]   = Int32(mend)
        var t_tris = Int32(0)
        for mi in range(mstart, mend):
            t_tris += Int32(len(s[unsafe_offset=0].meshes[mi].face_idxs))
        var t_bounds = unsafe_alloc[Float32](max(Int(t_tris), 1) * 6)
        var t_mesh   = unsafe_alloc[Int32](max(Int(t_tris), 1))
        var t_local  = unsafe_alloc[Int32](max(Int(t_tris), 1))
        var t_flat = Int32(0)
        for mi in range(mstart, mend):
            var pts = out_pts[unsafe_offset=mi]
            var vis = out_vis[unsafe_offset=mi]
            var nt  = Int(out_nt[unsafe_offset=mi])
            for ti in range(nt):
                var v0 = Int(vis[unsafe_offset=ti*3+0]) * 4
                var v1 = Int(vis[unsafe_offset=ti*3+1]) * 4
                var v2 = Int(vis[unsafe_offset=ti*3+2]) * 4
                var x0 = pts[unsafe_offset=v0]; var y0 = pts[unsafe_offset=v0+1]; var z0 = pts[unsafe_offset=v0+2]
                var x1 = pts[unsafe_offset=v1]; var y1 = pts[unsafe_offset=v1+1]; var z1 = pts[unsafe_offset=v1+2]
                var x2 = pts[unsafe_offset=v2]; var y2 = pts[unsafe_offset=v2+1]; var z2 = pts[unsafe_offset=v2+2]
                var tb = Int(t_flat) * 6
                t_bounds[unsafe_offset=tb+0] = min(x0, min(x1, x2))
                t_bounds[unsafe_offset=tb+1] = min(y0, min(y1, y2))
                t_bounds[unsafe_offset=tb+2] = min(z0, min(z1, z2))
                t_bounds[unsafe_offset=tb+3] = max(x0, max(x1, x2))
                t_bounds[unsafe_offset=tb+4] = max(y0, max(y1, y2))
                t_bounds[unsafe_offset=tb+5] = max(z0, max(z1, z2))
                t_mesh[unsafe_offset=Int(t_flat)]  = Int32(mi)
                t_local[unsafe_offset=Int(t_flat)] = Int32(ti)
                t_flat += 1
        var t_max_nodes = max(Int(t_tris) * 2 + 4, 1)
        var t_nodes = unsafe_alloc[BVH2Node](t_max_nodes)
        var t_order = unsafe_alloc[Int32](max(Int(t_tris), 1))
        var t_node_count = build_bvh2(t_bounds, t_tris, t_nodes, t_order)
        t_bounds.unsafe_free()
        var t_prim_ids = unsafe_alloc[PrimId_C](max(Int(t_tris), 1))
        for k in range(Int(t_tris)):
            var orig = Int(t_order[unsafe_offset=k])
            var mi = Int(t_mesh[unsafe_offset=orig])
            var ti = Int(t_local[unsafe_offset=orig])
            var mat_idx = Int(s[unsafe_offset=0].meshes[mi].mat_idx)
            var mat_idx_r = mat_idx if mat_idx >= 0 else Int(default_mat_idx)
            t_prim_ids[unsafe_offset=k] = PrimId_C(Int64(mi), Int64(ti * 3), Int64(mat_idx_r), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        t_mesh.unsafe_free(); t_local.unsafe_free(); t_order.unsafe_free()
        blas_nodes_arr[unsafe_offset=tmpl]   = t_nodes
        blas_primids_arr[unsafe_offset=tmpl] = t_prim_ids
        blas_node_counts[unsafe_offset=tmpl]   = t_node_count
        blas_primid_counts[unsafe_offset=tmpl] = t_tris

    var instances_c = unsafe_alloc[Instance_C](max(Int(total_instances), 1))
    for k in range(Int(total_instances)):
        var tmpl_idx = Int(s[unsafe_offset=0].instance_template_idx[k])
        var o2w = SIMD[DType.float32, 16](0.0)
        var w2o = SIMD[DType.float32, 16](0.0)
        for ci in range(16):
            o2w[ci] = s[unsafe_offset=0].instance_obj_to_world[k*16+ci]
            w2o[ci] = s[unsafe_offset=0].instance_world_to_obj[k*16+ci]
        instances_c[unsafe_offset=k] = Instance_C(o2w, w2o, Int32(tmpl_idx))

        # World-space AABB for the TLAS leaf: transform the BLAS root's 8 corners.
        var root = blas_nodes_arr[unsafe_offset=tmpl_idx][unsafe_offset=0]
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
        prim_bounds[unsafe_offset=ib+0] = wxmin; prim_bounds[unsafe_offset=ib+1] = wymin; prim_bounds[unsafe_offset=ib+2] = wzmin
        prim_bounds[unsafe_offset=ib+3] = wxmax; prim_bounds[unsafe_offset=ib+4] = wymax; prim_bounds[unsafe_offset=ib+5] = wzmax

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
    var bvh_nodes_gpu = unsafe_alloc[BVH2Node](max_bvh_nodes_gpu)
    var bvh_order_gpu = unsafe_alloc[Int32](Int(total_prims_gpu))
    var node_count_gpu = build_bvh2(prim_bounds, total_prims_gpu, bvh_nodes_gpu, bvh_order_gpu)

    # ---- CPU-inclusive TLAS: tris + curves + instances ----
    # Without instances this covers exactly the primitives above, in the same
    # order, so a second build would reproduce that tree node for node (and its
    # PrimIds entry for entry). Share those arrays instead; free_parsed_scene
    # frees shared arrays once.
    var shared_tlas = total_instances == 0
    var bvh_nodes = bvh_nodes_gpu
    var bvh_order = bvh_order_gpu
    var node_count = node_count_gpu
    if not shared_tlas:
        bvh_nodes = unsafe_alloc[BVH2Node](Int(total_prims) * 2 + 4)
        bvh_order = unsafe_alloc[Int32](Int(total_prims))
        node_count = build_bvh2(prim_bounds, total_prims, bvh_nodes, bvh_order)

    prim_bounds.unsafe_free()

    var prim_ids_gpu = unsafe_alloc[PrimId_C](Int(total_prims_gpu))
    var prim_ids = prim_ids_gpu
    if not shared_tlas:
        prim_ids = unsafe_alloc[PrimId_C](Int(total_prims))

    var mesh_al_idx = unsafe_alloc[Int32](max(n_meshes, 1))
    var running_al = Int32(0)
    for mi in range(n_meshes):
        if s[unsafe_offset=0].meshes[mi].is_area_light:
            mesh_al_idx[unsafe_offset=mi] = running_al
            running_al += 1
        else:
            mesh_al_idx[unsafe_offset=mi] = Int32(-1)

    # GPU-safe PrimId assignment — tris + curves only (no instance branch;
    # `orig` here is always < total_tris + total_curve_groups). Same
    # tri/curve logic as the CPU-inclusive loop below.
    for k in range(Int(total_prims_gpu)):
        var orig = Int(bvh_order_gpu[unsafe_offset=k])
        if orig < Int(total_tris):
            var mi = Int(tri_mesh[unsafe_offset=orig])
            var ti = Int(tri_local[unsafe_offset=orig])
            if s[unsafe_offset=0].meshes[mi].is_area_light:
                var al_idx = Int(mesh_al_idx[unsafe_offset=mi])
                prim_ids_gpu[unsafe_offset=k].type          = Int8(3)
                prim_ids_gpu[unsafe_offset=k].id1           = Int64(al_idx)
                prim_ids_gpu[unsafe_offset=k].id2           = (Int64(mi) << 32) | Int64(ti)
                prim_ids_gpu[unsafe_offset=k].materialIndex = Int64(al_mat_base + al_idx)
            else:
                var mat_idx = Int(s[unsafe_offset=0].meshes[mi].mat_idx)
                prim_ids_gpu[unsafe_offset=k].type          = Int8(0)
                prim_ids_gpu[unsafe_offset=k].id1           = Int64(mi)
                prim_ids_gpu[unsafe_offset=k].id2           = Int64(ti * 3)
                prim_ids_gpu[unsafe_offset=k].materialIndex = Int64(mat_idx) if mat_idx >= 0 else Int64(default_mat_idx)
        else:
            var gidx = orig - Int(total_tris)
            var ci = Int(group_curve_idx[unsafe_offset=gidx])
            var curve_mat_idx = Int(s[unsafe_offset=0].curves_mat[ci])
            prim_ids_gpu[unsafe_offset=k].type          = Int8(5)
            prim_ids_gpu[unsafe_offset=k].id1           = Int64(ci)
            prim_ids_gpu[unsafe_offset=k].id2           = Int64(group_id2[unsafe_offset=gidx])
            prim_ids_gpu[unsafe_offset=k].materialIndex = Int64(curve_al_mat_idx[unsafe_offset=ci]) if s[unsafe_offset=0].curves_al[ci] else (Int64(curve_mat_idx) if curve_mat_idx >= 0 else Int64(default_mat_idx))
        prim_ids_gpu[unsafe_offset=k].instanceIdx = Int32(-1)
        prim_ids_gpu[unsafe_offset=k]._pad0 = Int8(0); prim_ids_gpu[unsafe_offset=k]._pad1 = Int8(0); prim_ids_gpu[unsafe_offset=k]._pad2 = Int8(0)

    for k in range(0 if shared_tlas else Int(total_prims)):   # shared: filled above
        var orig = Int(bvh_order[unsafe_offset=k])
        if orig < Int(total_tris):
            var mi   = Int(tri_mesh[unsafe_offset=orig])
            var ti   = Int(tri_local[unsafe_offset=orig])
            if s[unsafe_offset=0].meshes[mi].is_area_light:
                var al_idx = Int(mesh_al_idx[unsafe_offset=mi])
                prim_ids[unsafe_offset=k].type          = Int8(3)
                prim_ids[unsafe_offset=k].id1           = Int64(al_idx)
                prim_ids[unsafe_offset=k].id2           = (Int64(mi) << 32) | Int64(ti)
                prim_ids[unsafe_offset=k].materialIndex = Int64(al_mat_base + al_idx)
            else:
                var mat_idx = Int(s[unsafe_offset=0].meshes[mi].mat_idx)
                prim_ids[unsafe_offset=k].type          = Int8(0)
                prim_ids[unsafe_offset=k].id1           = Int64(mi)
                prim_ids[unsafe_offset=k].id2           = Int64(ti * 3)
                prim_ids[unsafe_offset=k].materialIndex = Int64(mat_idx) if mat_idx >= 0 else Int64(default_mat_idx)
        elif orig < Int(total_tris) + Int(total_curve_groups):
            var gidx = orig - Int(total_tris)
            var ci = Int(group_curve_idx[unsafe_offset=gidx])
            var curve_mat_idx = Int(s[unsafe_offset=0].curves_mat[ci])
            prim_ids[unsafe_offset=k].type          = Int8(5)
            prim_ids[unsafe_offset=k].id1           = Int64(ci)
            prim_ids[unsafe_offset=k].id2           = Int64(group_id2[unsafe_offset=gidx])
            prim_ids[unsafe_offset=k].materialIndex = Int64(curve_al_mat_idx[unsafe_offset=ci]) if s[unsafe_offset=0].curves_al[ci] else (Int64(curve_mat_idx) if curve_mat_idx >= 0 else Int64(default_mat_idx))
        else:
            var inst_idx = orig - Int(total_tris) - Int(total_curve_groups)
            prim_ids[unsafe_offset=k].type          = Int8(6)
            prim_ids[unsafe_offset=k].id1           = Int64(inst_idx)
            prim_ids[unsafe_offset=k].id2           = Int64(0)
            prim_ids[unsafe_offset=k].materialIndex = Int64(0)
        # instanceIdx is only meaningful on a *resolved* triangle hit (set by
        # traverse_bvh2_core's type==6 branch when it recurses into a BLAS) —
        # every ordinary top-level entry here, instance leaves included,
        # starts at -1.
        prim_ids[unsafe_offset=k].instanceIdx = Int32(-1)
        prim_ids[unsafe_offset=k]._pad0 = Int8(0); prim_ids[unsafe_offset=k]._pad1 = Int8(0)
        prim_ids[unsafe_offset=k]._pad2 = Int8(0)

    if not shared_tlas:
        bvh_order.unsafe_free()
    tri_mesh.unsafe_free(); tri_local.unsafe_free(); bvh_order_gpu.unsafe_free(); mesh_al_idx.unsafe_free()
    curve_al_mat_idx.unsafe_free()
    curve_group_base.unsafe_free(); group_curve_idx.unsafe_free(); group_id2.unsafe_free()

    # ---- Sampler params ----
    var spp = s[unsafe_offset=0].samples_per_pixel
    var log2_spp = Int32(0)
    var tmp_spp = spp
    while tmp_spp > Int32(1):
        tmp_spp >>= 1
        log2_spp += 1
    var log4_spp = (log2_spp + Int32(1)) / Int32(2)
    var dim = max(s[unsafe_offset=0].film_w, s[unsafe_offset=0].film_h)
    var log2_dim = Int32(0)
    var tmp_dim = dim
    while tmp_dim > Int32(1):
        tmp_dim >>= 1
        log2_dim += 1
    var n_base4 = log2_dim + log4_spp

    # ---- Filter norms ----
    var norm_x = gaussian_norm(s[unsafe_offset=0].filter_support_x, s[unsafe_offset=0].filter_sigma)
    var norm_y = gaussian_norm(s[unsafe_offset=0].filter_support_y, s[unsafe_offset=0].filter_sigma)
    var fweight = Float32(1.0)

    # ---- RNG seed from time ----
    var rng_seed = UInt64(perf_counter_ns())

    # ---- Film filename copy ----
    var fname = unsafe_alloc[UInt8](PSC_FILE_MAX)
    var fnstr = s[unsafe_offset=0].film_filename
    var fnlen = min(fnstr.byte_length(), PSC_FILE_MAX - 1)
    for fi in range(fnlen): fname[unsafe_offset=fi] = fnstr.unsafe_ptr()[unsafe_offset=fi]
    fname[unsafe_offset=fnlen] = UInt8(0)

    # ---- Texture filename table ----
    var n_tex = len(s[unsafe_offset=0].tex_names)
    var tex_ptrs = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](max(n_tex, 1))
    for ti in range(n_tex):
        var fstr = s[unsafe_offset=0].tex_files[ti]
        var slen = fstr.byte_length()
        var copy = unsafe_alloc[UInt8](slen + 1)
        for ci in range(slen): copy[unsafe_offset=ci] = fstr.unsafe_ptr()[unsafe_offset=ci]
        copy[unsafe_offset=slen] = UInt8(0)
        tex_ptrs[unsafe_offset=ti] = copy

    # ---- Fill output struct ----
    psc[unsafe_offset=0].materials        = mats
    psc[unsafe_offset=0].material_count   = Int32(n_mats)
    psc[unsafe_offset=0].area_lights      = al_list
    psc[unsafe_offset=0].area_light_count = al_count
    psc[unsafe_offset=0].meshes           = meshes
    psc[unsafe_offset=0].mesh_pts         = out_pts
    psc[unsafe_offset=0].mesh_vis         = out_vis
    psc[unsafe_offset=0].mesh_fis         = out_fis
    psc[unsafe_offset=0].mesh_n_verts     = out_nv
    psc[unsafe_offset=0].mesh_n_tris      = out_nt
    psc[unsafe_offset=0].mesh_uv_n_verts  = out_uv_nv
    psc[unsafe_offset=0].mesh_nrm_n_verts = out_nrm_nv
    psc[unsafe_offset=0].mesh_count       = Int32(n_meshes)
    psc[unsafe_offset=0].bvh_nodes        = bvh_nodes_gpu
    psc[unsafe_offset=0].prim_ids         = prim_ids_gpu
    psc[unsafe_offset=0].bvh_node_count   = node_count_gpu
    psc[unsafe_offset=0].prim_count       = total_prims_gpu
    psc[unsafe_offset=0].bvh_nodes_cpu      = bvh_nodes
    psc[unsafe_offset=0].prim_ids_cpu       = prim_ids
    psc[unsafe_offset=0].bvh_node_count_cpu = node_count
    psc[unsafe_offset=0].prim_count_cpu     = total_prims
    psc[unsafe_offset=0].blas_nodes_arr   = blas_nodes_arr
    psc[unsafe_offset=0].blas_primids_arr = blas_primids_arr
    psc[unsafe_offset=0].blas_node_counts   = blas_node_counts
    psc[unsafe_offset=0].blas_primid_counts = blas_primid_counts
    psc[unsafe_offset=0].blas_count       = Int32(n_templates)
    psc[unsafe_offset=0].instances        = instances_c
    psc[unsafe_offset=0].instance_count   = total_instances
    psc[unsafe_offset=0].template_mesh_start = template_mesh_start
    psc[unsafe_offset=0].template_mesh_end   = template_mesh_end
    psc[unsafe_offset=0].film_w           = s[unsafe_offset=0].film_w
    psc[unsafe_offset=0].film_h           = s[unsafe_offset=0].film_h
    psc[unsafe_offset=0].crop_x0          = s[unsafe_offset=0].crop_x0
    psc[unsafe_offset=0].crop_x1          = s[unsafe_offset=0].crop_x1
    psc[unsafe_offset=0].crop_y0          = s[unsafe_offset=0].crop_y0
    psc[unsafe_offset=0].crop_y1          = s[unsafe_offset=0].crop_y1
    psc[unsafe_offset=0].camera_fov       = s[unsafe_offset=0].camera_fov
    psc[unsafe_offset=0].film_iso         = s[unsafe_offset=0].film_iso
    psc[unsafe_offset=0].film_exposuretime = s[unsafe_offset=0].film_exposuretime
    psc[unsafe_offset=0].film_wb          = _film_white_balance_matrix(s[unsafe_offset=0].film_whitebalance)
    psc[unsafe_offset=0].film_max_comp    = s[unsafe_offset=0].film_max_comp
    psc[unsafe_offset=0].film_filename    = fname
    psc[unsafe_offset=0].filter_sigma     = s[unsafe_offset=0].filter_sigma
    psc[unsafe_offset=0].filter_support_x = s[unsafe_offset=0].filter_support_x
    psc[unsafe_offset=0].filter_support_y = s[unsafe_offset=0].filter_support_y
    psc[unsafe_offset=0].filter_type      = s[unsafe_offset=0].filter_type
    psc[unsafe_offset=0].filter_norm_x    = norm_x
    psc[unsafe_offset=0].filter_norm_y    = norm_y
    psc[unsafe_offset=0].filter_weight    = fweight
    psc[unsafe_offset=0].camera_fov       = s[unsafe_offset=0].camera_fov
    psc[unsafe_offset=0].samples_per_pixel = spp
    psc[unsafe_offset=0].log2_spp         = log2_spp
    psc[unsafe_offset=0].n_base4_digits   = n_base4
    psc[unsafe_offset=0].max_depth        = s[unsafe_offset=0].max_depth
    psc[unsafe_offset=0].rng_seed         = rng_seed
    psc[unsafe_offset=0].sppm_radius           = s[unsafe_offset=0].sppm_radius
    psc[unsafe_offset=0].sppm_photons_per_iter = s[unsafe_offset=0].sppm_photons_per_iter
    psc[unsafe_offset=0].tex_filenames    = tex_ptrs
    psc[unsafe_offset=0].tex_count        = Int32(n_tex)

    # ---- Normal maps, converted once into LEAN slope space ----
    # Only the SMS/MNEE manifold walk reads these; ordinary shading samples
    # the same file through the usual texture path. The walk additionally
    # needs the normal's analytic derivatives, which bilinear interpolation
    # of the raw RGB does not give consistently with the reference -- see
    # geometry.mojo's NormalSlopeMap_C for the representation and
    # sms.mojo's nmap_eval/nmap_eval_derivs for the evaluation.
    psc[unsafe_offset=0].nmaps = Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling()
    if n_tex > 0:
        var nmaps = unsafe_alloc[NormalSlopeMap_C](n_tex)
        for ti in range(n_tex):
            nmaps[unsafe_offset=ti] = normal_slope_map_none()
        # Each map is decoded and converted independently, and a scene can have
        # many (Bistro: 132, ~15 s of startup serially), so convert them on all
        # cores. First collect the distinct texture indices, in material order.
        var is_nmap = unsafe_alloc[Bool](n_tex)
        for ti in range(n_tex):
            is_nmap[unsafe_offset=ti] = False
        var nm_idx = unsafe_alloc[Int](n_tex)
        var n_nm = 0
        for mi in range(n_mats):
            var nti = Int(mats[unsafe_offset=mi].normal_tex_idx)
            if nti >= 0 and nti < n_tex and not is_nmap[unsafe_offset=nti]:
                is_nmap[unsafe_offset=nti] = True
                nm_idx[unsafe_offset=n_nm] = nti
                n_nm += 1
        # (w, h) of each map skipped for not being square, reported below in order.
        var nonsquare = unsafe_alloc[Int32](max(n_nm, 1) * 2)
        var next_nm = unsafe_alloc[Int32](1)
        next_nm[unsafe_offset=0] = Int32(0)

        def nmap_worker(_worker_idx: Int) {imm}:
            while True:
                var k = Int(Atomic.fetch_add(next_nm, Int32(1)))
                if k >= n_nm:
                    break
                var nti = nm_idx[unsafe_offset=k]
                nonsquare[unsafe_offset=k * 2] = Int32(0); nonsquare[unsafe_offset=k * 2 + 1] = Int32(0)
                var np_ptr = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
                var nw_out = unsafe_alloc[Int32](1); var nh_out = unsafe_alloc[Int32](1)
                nw_out[unsafe_offset=0] = Int32(0); nh_out[unsafe_offset=0] = Int32(0)
                var nm_ok = external_call["load_texture_rgb", Int32,
                    Pointer[UInt8, MutUntrackedOrigin],
                    Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
                    Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
                    Int32](
                    tex_ptrs[unsafe_offset=nti], np_ptr, nw_out, nh_out, Int32(1))   # raw=1: no sRGB decode
                var nw = Int(nw_out[unsafe_offset=0]); var nh = Int(nh_out[unsafe_offset=0])
                nw_out.unsafe_free(); nh_out.unsafe_free()
                if nm_ok != Int32(0) and nw > 0 and nw == nh:
                    var src = np_ptr[unsafe_offset=0]
                    var slopes = unsafe_alloc[Float32](2 * nw * nh)
                    for i in range(nw * nh):
                        var nx = Float32(2.0)*src[unsafe_offset=i*3+0] - Float32(1.0)
                        var ny = Float32(2.0)*src[unsafe_offset=i*3+1] - Float32(1.0)
                        var nz = Float32(2.0)*src[unsafe_offset=i*3+2] - Float32(1.0)
                        var ln = sqrt(nx*nx + ny*ny + nz*nz)
                        if ln > Float32(1e-8) and abs(nz) > Float32(1e-6):
                            slopes[unsafe_offset=i*2+0] = -nx / nz
                            slopes[unsafe_offset=i*2+1] = -ny / nz
                        else:
                            slopes[unsafe_offset=i*2+0] = Float32(0.0)
                            slopes[unsafe_offset=i*2+1] = Float32(0.0)
                    nmaps[unsafe_offset=nti] = NormalSlopeMap_C(slopes, Int32(nw))
                    _ = external_call["free_texture_rgb", Int32,
                        Pointer[Float32, MutUntrackedOrigin]](src)
                elif nm_ok != Int32(0) and nw > 0:
                    # Non-square: the slope-map addressing (and the reference it
                    # mirrors) assumes square, power-of-two maps. Leave res == 0
                    # so the walk falls back to the smooth surface rather than
                    # reading the map with the wrong stride.
                    nonsquare[unsafe_offset=k * 2] = Int32(nw); nonsquare[unsafe_offset=k * 2 + 1] = Int32(nh)
                    _ = external_call["free_texture_rgb", Int32,
                        Pointer[Float32, MutUntrackedOrigin]](np_ptr[unsafe_offset=0])
                np_ptr.unsafe_free()

        if n_nm > 0:
            parallelize(nmap_worker, min(num_performance_cores(), n_nm))
        for k in range(n_nm):
            if nonsquare[unsafe_offset=k * 2] > Int32(0):
                print("warning: normal map is not square (", Int(nonsquare[unsafe_offset=k * 2]), "x", Int(nonsquare[unsafe_offset=k * 2 + 1]),
                      "), SMS manifold walk will treat this surface as smooth")
        is_nmap.unsafe_free(); nm_idx.unsafe_free(); nonsquare.unsafe_free(); next_nm.unsafe_free()
        psc[unsafe_offset=0].nmaps = nmaps

    # ---- Non-area lights ----
    var nd = len(s[unsafe_offset=0].distant_dirs) // 3
    if nd > 0:
        var dl_buf = unsafe_alloc[DistantLight_C](nd)
        for i in range(nd):
            dl_buf[unsafe_offset=i] = DistantLight_C(
                Vec3f(s[unsafe_offset=0].distant_dirs[i*3+0], s[unsafe_offset=0].distant_dirs[i*3+1], s[unsafe_offset=0].distant_dirs[i*3+2]),
                Float32(0),
                RGB(s[unsafe_offset=0].distant_rgbs[i*3+0], s[unsafe_offset=0].distant_rgbs[i*3+1], s[unsafe_offset=0].distant_rgbs[i*3+2]),
                Float32(0))
        psc[unsafe_offset=0].distant_lights = dl_buf
    else:
        psc[unsafe_offset=0].distant_lights = Pointer[DistantLight_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].distant_count = Int32(nd)

    var np2 = len(s[unsafe_offset=0].point_pos) // 3
    if np2 > 0:
        var pl_buf = unsafe_alloc[PointLight_C](np2)
        for i in range(np2):
            pl_buf[unsafe_offset=i] = PointLight_C(
                Point3f(s[unsafe_offset=0].point_pos[i*3+0], s[unsafe_offset=0].point_pos[i*3+1], s[unsafe_offset=0].point_pos[i*3+2]),
                Float32(0),
                RGB(s[unsafe_offset=0].point_rgbs[i*3+0], s[unsafe_offset=0].point_rgbs[i*3+1], s[unsafe_offset=0].point_rgbs[i*3+2]),
                Float32(0))
        psc[unsafe_offset=0].point_lights = pl_buf
    else:
        psc[unsafe_offset=0].point_lights = Pointer[PointLight_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].point_count = Int32(np2)

    var ni = len(s[unsafe_offset=0].inf_tex_idx)
    if ni > 0:
        var il_buf = unsafe_alloc[InfiniteLight_C](ni)
        for i in range(ni):
            var tidx = s[unsafe_offset=0].inf_tex_idx[i]
            var sc = RGB(s[unsafe_offset=0].inf_rgb[i*3+0], s[unsafe_offset=0].inf_rgb[i*3+1], s[unsafe_offset=0].inf_rgb[i*3+2])
            var cdf_w = Int32(0); var cdf_h = Int32(0)
            var cdf_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            var raw_pixels = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            if tidx >= Int32(0):
                var fname2 = psc[unsafe_offset=0].tex_filenames[unsafe_offset=Int(tidx)]
                var pixels_ptr = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
                var iw_out = unsafe_alloc[Int32](1); var ih_out = unsafe_alloc[Int32](1)
                iw_out[unsafe_offset=0] = Int32(0); ih_out[unsafe_offset=0] = Int32(0)
                var load_ok = external_call["load_texture_rgb", Int32,
                    Pointer[UInt8, MutUntrackedOrigin],
                    Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
                    Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
                    Int32](
                    fname2, pixels_ptr, iw_out, ih_out, Int32(0))
                var iw = Int(iw_out[unsafe_offset=0]); var ih = Int(ih_out[unsafe_offset=0])
                iw_out.unsafe_free(); ih_out.unsafe_free()
                if load_ok != Int32(0) and iw > 0 and ih > 0:
                    var pixels = pixels_ptr[unsafe_offset=0]
                    # Do NOT vertically flip here (removed a flip added in
                    # 73f368fd "fix upside-down environment"): in this equal-area
                    # octahedral mapping, row-flip (v -> 1-v) mirrors the decoded
                    # direction's Y axis, not Z/elevation, so it wasn't a valid
                    # "upside-down" fix. Confirmed wrong via cast-shadow direction
                    # (bunny-fur) vs env map luminance centroid; see
                    # project_infinite_light_shadows memory.
                    raw_pixels = pixels
                    var cdf_size = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = unsafe_alloc[Float32](cdf_size)
                    var row_sums = unsafe_alloc[Float32](ih)
                    for ry in range(ih):
                        var row_sum = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            row_sum += lum
                        row_sums[unsafe_offset=ry] = row_sum
                    cdf_buf[unsafe_offset=0] = Float32(0.0)
                    for ry in range(ih):
                        cdf_buf[unsafe_offset=ry + 1] = cdf_buf[unsafe_offset=ry] + row_sums[unsafe_offset=ry]
                    var total = cdf_buf[unsafe_offset=ih]
                    if total > Float32(0.0):
                        var inv_total = Float32(1.0) / total
                        for ry in range(ih + 1):
                            cdf_buf[unsafe_offset=ry] *= inv_total
                    for ry in range(ih):
                        var base = (ih + 1) + ry * (iw + 1)
                        cdf_buf[unsafe_offset=base] = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[unsafe_offset=(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            cdf_buf[unsafe_offset=base + rx + 1] = cdf_buf[unsafe_offset=base + rx] + lum
                        var row_total = cdf_buf[unsafe_offset=base + iw]
                        if row_total > Float32(0.0):
                            var inv_rt = Float32(1.0) / row_total
                            for rx in range(iw + 1):
                                cdf_buf[unsafe_offset=base + rx] *= inv_rt
                    row_sums.unsafe_free()
                    cdf_ptr = cdf_buf
                    cdf_w = Int32(iw); cdf_h = Int32(ih)
                pixels_ptr.unsafe_free()
            var w2l = unsafe_alloc[Float32](16)
            var light_ctm_base = i * 16
            var is_identity = True
            for ci in range(16):
                var expected = Float32(1) if (ci == 0 or ci == 5 or ci == 10 or ci == 15) else Float32(0)
                if s[unsafe_offset=0].inf_ctm[light_ctm_base + ci] != expected:
                    is_identity = False
                    break
            if is_identity:
                _psc_identity(w2l)
            else:
                var light_ctm_tmp = unsafe_alloc[Float32](16)
                for ci in range(16): light_ctm_tmp[unsafe_offset=ci] = s[unsafe_offset=0].inf_ctm[light_ctm_base + ci]
                _ = matrix_invert(light_ctm_tmp, w2l)
                light_ctm_tmp.unsafe_free()
            il_buf[unsafe_offset=i] = InfiniteLight_C(sc, tidx, cdf_w, cdf_h, cdf_ptr, raw_pixels, w2l)
        psc[unsafe_offset=0].infinite_lights = il_buf
    else:
        psc[unsafe_offset=0].infinite_lights = Pointer[InfiniteLight_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].infinite_count = Int32(ni)

    # ---- Analytical spheres ----
    var ns = len(s[unsafe_offset=0].spheres_cx)
    if ns > 0:
        var sph_buf = unsafe_alloc[Sphere_C](ns)
        for i in range(ns):
            var em = RGB(s[unsafe_offset=0].spheres_rgb[i].r, s[unsafe_offset=0].spheres_rgb[i].g, s[unsafe_offset=0].spheres_rgb[i].b)
            var al_flag = Int8(1) if s[unsafe_offset=0].spheres_al[i] else Int8(0)
            var sph_mat_idx = s[unsafe_offset=0].spheres_mat[i]
            if sph_mat_idx == Int32(-1) and not s[unsafe_offset=0].spheres_al[i]:
                sph_mat_idx = default_mat_idx
            sph_buf[unsafe_offset=i] = Sphere_C(
                Point3f(s[unsafe_offset=0].spheres_cx[i], s[unsafe_offset=0].spheres_cy[i], s[unsafe_offset=0].spheres_cz[i]),
                s[unsafe_offset=0].spheres_r[i],
                sph_mat_idx,
                al_flag,
                Int8(0), Int8(0), Int8(0),
                em)
        psc[unsafe_offset=0].spheres = sph_buf
    else:
        psc[unsafe_offset=0].spheres = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].sphere_count = Int32(ns)

    # ---- Native curves (hair/fur) ----
    # n_pieces per curve (flatness-adaptive) was already computed above for
    # the per-piece BVH leaf construction — reuse it here instead of
    # recomputing.
    var nc = len(s[unsafe_offset=0].curves_mat)
    if nc > 0:
        var curve_buf = unsafe_alloc[Curve_C](nc)
        for i in range(nc):
            var cb = i * 12
            curve_buf[unsafe_offset=i] = Curve_C(
                Point3f(s[unsafe_offset=0].curves_cp[cb+0], s[unsafe_offset=0].curves_cp[cb+1], s[unsafe_offset=0].curves_cp[cb+2]),
                Point3f(s[unsafe_offset=0].curves_cp[cb+3], s[unsafe_offset=0].curves_cp[cb+4], s[unsafe_offset=0].curves_cp[cb+5]),
                Point3f(s[unsafe_offset=0].curves_cp[cb+6], s[unsafe_offset=0].curves_cp[cb+7], s[unsafe_offset=0].curves_cp[cb+8]),
                Point3f(s[unsafe_offset=0].curves_cp[cb+9], s[unsafe_offset=0].curves_cp[cb+10], s[unsafe_offset=0].curves_cp[cb+11]),
                s[unsafe_offset=0].curves_w0[i], s[unsafe_offset=0].curves_w1[i], s[unsafe_offset=0].curves_mat[i], curve_n_pieces[unsafe_offset=i])
        psc[unsafe_offset=0].curves = curve_buf
    else:
        psc[unsafe_offset=0].curves = Pointer[Curve_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].curve_count = Int32(nc)
    curve_n_pieces.unsafe_free()

    # ---- Curve area light NEE entries (al_list[n_al_mesh:]) ----
    # Mirrors the mesh area-light loop above, but needs curve_buf/n_pieces
    # (curve_light_tube_area), which only exist from this point on.
    var cl_running = Int32(0)
    for i in range(nc):
        if s[unsafe_offset=0].curves_al[i]:
            var idx = n_al_mesh + Int(cl_running)
            al_list[unsafe_offset=idx].meshIdx    = Int32(i)
            al_list[unsafe_offset=idx].n_tris     = Int32(0)
            al_list[unsafe_offset=idx].emission   = s[unsafe_offset=0].curves_al_rgb[i]
            al_list[unsafe_offset=idx].total_area = curve_light_tube_area(psc[unsafe_offset=0].curves[unsafe_offset=i])
            al_list[unsafe_offset=idx].kind       = Int8(1)
            cl_running += 1
    psc[unsafe_offset=0].area_light_count = Int32(n_al_mesh) + cl_running

    # ---- Heterogeneous density grids ("uniformgrid" media) ----
    var ng = len(s[unsafe_offset=0].grid_nx)
    if ng > 0:
        var grid_buf = unsafe_alloc[Grid_C](ng)
        for i in range(ng):
            var nx = s[unsafe_offset=0].grid_nx[i]; var ny = s[unsafe_offset=0].grid_ny[i]; var nz = s[unsafe_offset=0].grid_nz[i]
            var n_voxels = Int(nx) * Int(ny) * Int(nz)
            var density_buf = unsafe_alloc[Float32](max(n_voxels, 1))
            var base = Int(s[unsafe_offset=0].grid_density_base[i])
            var max_d = Float32(0.0)
            for vi in range(n_voxels):
                var dv = s[unsafe_offset=0].grid_density[base + vi]
                density_buf[unsafe_offset=vi] = dv
                if dv > max_d: max_d = dv
            var ctm_tmp = unsafe_alloc[Float32](16)
            var w2m = unsafe_alloc[Float32](16)
            for ci in range(16):
                ctm_tmp[unsafe_offset=ci] = s[unsafe_offset=0].grid_ctm[i*16 + ci]
            _ = matrix_invert(ctm_tmp, w2m)
            var w2m_simd = SIMD[DType.float32, 16](
                w2m[unsafe_offset=0], w2m[unsafe_offset=1], w2m[unsafe_offset=2], w2m[unsafe_offset=3], w2m[unsafe_offset=4], w2m[unsafe_offset=5], w2m[unsafe_offset=6], w2m[unsafe_offset=7],
                w2m[unsafe_offset=8], w2m[unsafe_offset=9], w2m[unsafe_offset=10], w2m[unsafe_offset=11], w2m[unsafe_offset=12], w2m[unsafe_offset=13], w2m[unsafe_offset=14], w2m[unsafe_offset=15])
            grid_buf[unsafe_offset=i] = Grid_C(
                density_buf, nx, ny, nz,
                Point3f(s[unsafe_offset=0].grid_p0[i*3], s[unsafe_offset=0].grid_p0[i*3+1], s[unsafe_offset=0].grid_p0[i*3+2]),
                Point3f(s[unsafe_offset=0].grid_p1[i*3], s[unsafe_offset=0].grid_p1[i*3+1], s[unsafe_offset=0].grid_p1[i*3+2]),
                w2m_simd, max_d)
            ctm_tmp.unsafe_free(); w2m.unsafe_free()
        psc[unsafe_offset=0].grids = grid_buf
    else:
        psc[unsafe_offset=0].grids = Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].grid_count = Int32(ng)

    # ---- Sparse density grids ("nanovdb" media) ----
    # The .nvdb files themselves are loaded HERE, not at parse time (see
    # SceneParseState.nvdb_filenames's own comment) -- one C-bridge call per
    # grid, decompressing the whole blob (ZIP, for every real asset this
    # project has seen: bunny_cloud packs 146.6MB of grid into 76MB on
    # disk). A failed load (bad path, unsupported grid type -- the bridge
    # already refuses non-Float grids) leaves that medium with an empty
    # blob and a zero majorant, which nvdb_sample_density's index-bounds
    # check turns into "always returns background" rather than a crash --
    # silently wrong-looking (a missing cloud renders as empty air) but
    # never unsafe, and `print`ed so it isn't silent in the log.
    var nvg = len(s[unsafe_offset=0].nvdb_filenames)
    if nvg > 0:
        var nvdb_buf = unsafe_alloc[NvdbGrid_C](nvg)
        for i in range(nvg):
            var path_str = s[unsafe_offset=0].nvdb_filenames[i]
            var plen = path_str.byte_length()
            var cpath = unsafe_alloc[UInt8](plen + 1)
            for ci in range(plen):
                cpath[unsafe_offset=ci] = path_str.unsafe_ptr()[unsafe_offset=ci]
            cpath[unsafe_offset=plen] = UInt8(0)
            var gname = s[unsafe_offset=0].nvdb_gridnames[i]
            var glen = gname.byte_length()
            var cname = unsafe_alloc[UInt8](glen + 1)
            for ci in range(glen):
                cname[unsafe_offset=ci] = gname.unsafe_ptr()[unsafe_offset=ci]
            cname[unsafe_offset=glen] = UInt8(0)
            var handle = nvdb_load_named(cpath, cname) if glen > 0 else nvdb_load(cpath, Int32(0))
            # A named lookup that misses is normal for the density grid too:
            # some .nvdb files carry a single UNNAMED grid, so fall back to
            # "first grid by index" rather than failing the whole medium.
            if Int(handle) == 0 and glen > 0 and gname == "density":
                handle = nvdb_load(cpath, Int32(0))
            cname.unsafe_free()
            cpath.unsafe_free()

            var blob: Pointer[UInt8, MutUntrackedOrigin]
            var blob_size_v = Int64(0)
            var idx_min = Point3f(Float32(0), Float32(0), Float32(0))
            var idx_max = Point3f(Float32(-1), Float32(-1), Float32(-1))  # empty range: min > max
            var max_d = Float32(0.0)
            var imat = SIMD[DType.float32, 16](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            var mvec = Vec3f(Float32(0), Float32(0), Float32(0))
            if Int(handle) == 0:
                # Expected and silent for a "temperature" grid the file does
                # not contain (a non-emissive volume); only a missing DENSITY
                # grid is worth warning about.
                if s[unsafe_offset=0].nvdb_gridnames[i] != "temperature":
                    print("Warning: could not load nanovdb grid '" + s[unsafe_offset=0].nvdb_gridnames[i] + "' from:", path_str)
                blob = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
            else:
                var blob_size = Int(nvdb_size(handle))
                blob_size_v = Int64(blob_size)
                blob = unsafe_alloc[UInt8](max(blob_size, 1))
                var src = nvdb_data(handle)
                # memcpy, not a per-byte Mojo loop: bunny_cloud alone is
                # 146.6MB decompressed, and a scalar byte-index loop over
                # that is orders of magnitude slower than a real memcpy --
                # slow enough it looked like a hang/crash during bring-up.
                unsafe_memcpy(dest=blob, src=src, count=blob_size)
                var ibbmin = unsafe_alloc[Int32](3); var ibbmax = unsafe_alloc[Int32](3)
                nvdb_index_bbox(handle, ibbmin, ibbmax)
                idx_min = Point3f(Float32(ibbmin[unsafe_offset=0]), Float32(ibbmin[unsafe_offset=1]), Float32(ibbmin[unsafe_offset=2]))
                idx_max = Point3f(Float32(ibbmax[unsafe_offset=0]), Float32(ibbmax[unsafe_offset=1]), Float32(ibbmax[unsafe_offset=2]))
                ibbmin.unsafe_free(); ibbmax.unsafe_free()
                var lo = unsafe_alloc[Float32](1); var hi = unsafe_alloc[Float32](1)
                lo[unsafe_offset=0] = Float32(0); hi[unsafe_offset=0] = Float32(0)
                nvdb_value_range(handle, lo, hi)
                max_d = hi[unsafe_offset=0]
                lo.unsafe_free(); hi.unsafe_free()
                var im9 = unsafe_alloc[Float32](9); var v3 = unsafe_alloc[Float32](3)
                nvdb_map_invmatf(handle, im9); nvdb_map_vecf(handle, v3)
                imat = SIMD[DType.float32, 16](
                    im9[unsafe_offset=0], im9[unsafe_offset=1], im9[unsafe_offset=2], im9[unsafe_offset=3], im9[unsafe_offset=4], im9[unsafe_offset=5], im9[unsafe_offset=6], im9[unsafe_offset=7], im9[unsafe_offset=8],
                    Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
                mvec = Vec3f(v3[unsafe_offset=0], v3[unsafe_offset=1], v3[unsafe_offset=2])
                im9.unsafe_free(); v3.unsafe_free()
                nvdb_free(handle)

            var ctm_tmp2 = unsafe_alloc[Float32](16)
            var w2m2 = unsafe_alloc[Float32](16)
            for ci in range(16):
                ctm_tmp2[unsafe_offset=ci] = s[unsafe_offset=0].nvdb_ctm[i*16 + ci]
            _ = matrix_invert(ctm_tmp2, w2m2)
            var w2m2_simd = SIMD[DType.float32, 16](
                w2m2[unsafe_offset=0], w2m2[unsafe_offset=1], w2m2[unsafe_offset=2], w2m2[unsafe_offset=3], w2m2[unsafe_offset=4], w2m2[unsafe_offset=5], w2m2[unsafe_offset=6], w2m2[unsafe_offset=7],
                w2m2[unsafe_offset=8], w2m2[unsafe_offset=9], w2m2[unsafe_offset=10], w2m2[unsafe_offset=11], w2m2[unsafe_offset=12], w2m2[unsafe_offset=13], w2m2[unsafe_offset=14], w2m2[unsafe_offset=15])
            ctm_tmp2.unsafe_free(); w2m2.unsafe_free()

            nvdb_buf[unsafe_offset=i] = NvdbGrid_C(blob, blob_size_v, w2m2_simd, imat, mvec, idx_min, idx_max, max_d)
        psc[unsafe_offset=0].nvdb_grids = nvdb_buf
    else:
        psc[unsafe_offset=0].nvdb_grids = Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].nvdb_grid_count = Int32(nvg)

    # ---- Media ----
    var nm = len(s[unsafe_offset=0].med_g)
    if nm > 0:
        var med_buf = unsafe_alloc[Medium_C](nm)
        for i in range(nm):
            var sa = RGB(s[unsafe_offset=0].med_sa[i*3], s[unsafe_offset=0].med_sa[i*3+1], s[unsafe_offset=0].med_sa[i*3+2])
            var ss = RGB(s[unsafe_offset=0].med_ss[i*3], s[unsafe_offset=0].med_ss[i*3+1], s[unsafe_offset=0].med_ss[i*3+2])
            med_buf[unsafe_offset=i] = Medium_C(sa, ss, s[unsafe_offset=0].med_g[i],
                                  s[unsafe_offset=0].med_grid_idx[i], s[unsafe_offset=0].med_nvdb_idx[i],
                                  s[unsafe_offset=0].med_nvdb_temp_idx[i], s[unsafe_offset=0].med_le_scale[i],
                                  s[unsafe_offset=0].med_temp_offset[i], s[unsafe_offset=0].med_temp_scale[i],
                                  s[unsafe_offset=0].med_is_sss[i])
        psc[unsafe_offset=0].mediums = med_buf
    else:
        psc[unsafe_offset=0].mediums = Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling()
    psc[unsafe_offset=0].medium_count = Int32(nm)

    # ---- Build power-weighted area light CDF ----
    var ls_n = Int(psc[unsafe_offset=0].area_light_count)
    var ls_cdf = unsafe_alloc[Float32](max(ls_n + 1, 2))
    ls_cdf[unsafe_offset=0] = Float32(0.0)
    var ls_total_power = Float32(0.0)
    for i in range(ls_n):
        var al = psc[unsafe_offset=0].area_lights[unsafe_offset=i]
        var power = al.emission.luma() * al.total_area
        ls_total_power += power
        ls_cdf[unsafe_offset=i + 1] = ls_cdf[unsafe_offset=i] + power
    if ls_total_power > Float32(0.0):
        var inv = Float32(1.0) / ls_total_power
        for i in range(1, ls_n + 1):
            ls_cdf[unsafe_offset=i] *= inv
    else:
        for i in range(1, ls_n + 1):
            ls_cdf[unsafe_offset=i] = Float32(i) / Float32(max(ls_n, 1))
    psc[unsafe_offset=0].light_sampler = LightSampler_C(ls_cdf, Int32(ls_n), Int32(0))

# ── Exported API ──────────────────────────────────────────────────────────────

def resize_film(psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
               new_w: Int32, new_h: Int32):
    psc[unsafe_offset=0].film_w = new_w
    psc[unsafe_offset=0].film_h = new_h

    var frame = Float32(new_w) / Float32(new_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = unsafe_alloc[Float32](16)
    make_screen_to_raster(new_w, new_h, smin_x, smax_x, smin_y, smax_y, str_mat)
    var rts = unsafe_alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)
    var cts = unsafe_alloc[Float32](16)
    make_perspective_matrix(psc[unsafe_offset=0].camera_fov, Float32(0.01), cts)
    var cts_inv = unsafe_alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)
    if Int(psc[unsafe_offset=0].raster_to_camera) > 1:
        psc[unsafe_offset=0].raster_to_camera.unsafe_free()
    var r2c = unsafe_alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c)
    psc[unsafe_offset=0].raster_to_camera = r2c
    cts.unsafe_free(); str_mat.unsafe_free(); rts.unsafe_free(); cts_inv.unsafe_free()

def mojo_parse_scene(path: Pointer[UInt8, MutUntrackedOrigin],
                     verbose: Bool = False,
                    ) -> Pointer[ParsedScene_Mojo, MutUntrackedOrigin]:
    external_call["createTextureSystem", NoneType]()
    var handle = scanner_open(path)
    if handle[unsafe_offset=0].is_at_end != Int32(0):
        print("Error: cannot open scene file:", String(unsafe_from_utf8_ptr=path.as_imm()))
        scanner_free(handle)
        return Pointer[ParsedScene_Mojo, MutUntrackedOrigin].unsafe_dangling()

    var s_ptr = unsafe_alloc[SceneParseState](1)
    s_ptr.unsafe_write(SceneParseState())
    var pi = 0
    while path[unsafe_offset=pi] != UInt8(0):
        pi += 1
    var last_slash = -1
    for ki in range(pi):
        if path[unsafe_offset=ki] == UInt8(47):
            last_slash = ki
    if last_slash >= 0:
        var dir_tmp = unsafe_alloc[UInt8](last_slash + 2)
        for ki in range(last_slash + 1):
            dir_tmp[unsafe_offset=ki] = path[unsafe_offset=ki]
        dir_tmp[unsafe_offset=last_slash + 1] = UInt8(0)
        s_ptr[unsafe_offset=0].scene_dir = String(unsafe_from_utf8_ptr=dir_tmp.as_imm())
        dir_tmp.unsafe_free()
    parse_scene_file(handle, s_ptr)
    scanner_free(handle)

    var psc = unsafe_alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc, verbose)
    _ = s_ptr.unsafe_take_pointee()
    s_ptr.unsafe_free()
    return psc

def mojo_parsed_free(psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin]):
    if Int(psc) == 0:
        return
    var n = Int(psc[unsafe_offset=0].mesh_count)
    for i in range(n):
        psc[unsafe_offset=0].mesh_pts[unsafe_offset=i].unsafe_free()
        psc[unsafe_offset=0].mesh_vis[unsafe_offset=i].unsafe_free()
        psc[unsafe_offset=0].mesh_fis[unsafe_offset=i].unsafe_free()
        if psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=i] > Int32(0):
            psc[unsafe_offset=0].meshes[unsafe_offset=i].uvs.unsafe_free()
        if psc[unsafe_offset=0].mesh_nrm_n_verts[unsafe_offset=i] > Int32(0):
            psc[unsafe_offset=0].meshes[unsafe_offset=i].normals.unsafe_free()
    if psc[unsafe_offset=0].mesh_count > 0:
        psc[unsafe_offset=0].mesh_pts.unsafe_free()
        psc[unsafe_offset=0].mesh_vis.unsafe_free()
        psc[unsafe_offset=0].mesh_fis.unsafe_free()
        psc[unsafe_offset=0].mesh_n_verts.unsafe_free()
        psc[unsafe_offset=0].mesh_n_tris.unsafe_free()
        psc[unsafe_offset=0].mesh_uv_n_verts.unsafe_free()
        psc[unsafe_offset=0].mesh_nrm_n_verts.unsafe_free()
        psc[unsafe_offset=0].meshes.unsafe_free()
    if psc[unsafe_offset=0].material_count > 0:
        psc[unsafe_offset=0].materials.unsafe_free()
    if psc[unsafe_offset=0].area_light_count > 0:
        psc[unsafe_offset=0].area_lights.unsafe_free()
    if psc[unsafe_offset=0].bvh_node_count > 0:
        psc[unsafe_offset=0].bvh_nodes.unsafe_free()
    if psc[unsafe_offset=0].prim_count > 0:
        psc[unsafe_offset=0].prim_ids.unsafe_free()
    # Without instances the CPU TLAS shares the GPU one's arrays (freed above).
    if psc[unsafe_offset=0].bvh_node_count_cpu > 0 and Int(psc[unsafe_offset=0].bvh_nodes_cpu) != Int(psc[unsafe_offset=0].bvh_nodes):
        psc[unsafe_offset=0].bvh_nodes_cpu.unsafe_free()
    if psc[unsafe_offset=0].prim_count_cpu > 0 and Int(psc[unsafe_offset=0].prim_ids_cpu) != Int(psc[unsafe_offset=0].prim_ids):
        psc[unsafe_offset=0].prim_ids_cpu.unsafe_free()
    if Int(psc[unsafe_offset=0].raster_to_camera) > 4:
        psc[unsafe_offset=0].raster_to_camera.unsafe_free()
    if Int(psc[unsafe_offset=0].camera_to_world) > 4:
        psc[unsafe_offset=0].camera_to_world.unsafe_free()
    if Int(psc[unsafe_offset=0].film_filename) > 1:
        psc[unsafe_offset=0].film_filename.unsafe_free()
    if psc[unsafe_offset=0].tex_count > 0:
        var nt = Int(psc[unsafe_offset=0].tex_count)
        for ti in range(nt):
            psc[unsafe_offset=0].tex_filenames[unsafe_offset=ti].unsafe_free()
        for ti in range(nt):
            if psc[unsafe_offset=0].nmaps[unsafe_offset=ti].res > Int32(0):
                psc[unsafe_offset=0].nmaps[unsafe_offset=ti].slopes.unsafe_free()
        psc[unsafe_offset=0].nmaps.unsafe_free()
        psc[unsafe_offset=0].tex_filenames.unsafe_free()
    if psc[unsafe_offset=0].distant_count > 0:
        psc[unsafe_offset=0].distant_lights.unsafe_free()
    if psc[unsafe_offset=0].point_count > 0:
        psc[unsafe_offset=0].point_lights.unsafe_free()
    if psc[unsafe_offset=0].infinite_count > 0:
        var ni = Int(psc[unsafe_offset=0].infinite_count)
        for ii in range(ni):
            var il = psc[unsafe_offset=0].infinite_lights[unsafe_offset=ii]
            if _is_real_ptr(il.cdf_ptr):
                il.cdf_ptr.unsafe_free()
            if _is_real_ptr(il.pixels_ptr):
                _ = external_call["free_texture_rgb", Int32,
                    Pointer[Float32, MutUntrackedOrigin]](il.pixels_ptr)
            il.world_to_light.unsafe_free()
        psc[unsafe_offset=0].infinite_lights.unsafe_free()
    if psc[unsafe_offset=0].sphere_count > 0:
        psc[unsafe_offset=0].spheres.unsafe_free()
    if psc[unsafe_offset=0].curve_count > 0:
        psc[unsafe_offset=0].curves.unsafe_free()
    if psc[unsafe_offset=0].measured_count > 0:
        # Per-pointer sentinel-address guards (matches light_sampler.cdf's
        # convention above): a MeasuredBRDF_C for a file that FAILED to load
        # has every pointer field set via unsafe_dangling() (see
        # measured_bsdf.mojo's _fail()), which must never be passed to
        # .unsafe_free() directly.
        for mi in range(Int(psc[unsafe_offset=0].measured_count)):
            var mb = psc[unsafe_offset=0].measured_brdfs[unsafe_offset=mi]
            if Int(mb.theta_i) > 4: mb.theta_i.unsafe_free()
            if Int(mb.phi_i) > 4: mb.phi_i.unsafe_free()
            if Int(mb.wavelengths) > 4: mb.wavelengths.unsafe_free()
            if Int(mb.ndf_data) > 4: mb.ndf_data.unsafe_free()
            if Int(mb.sigma_data) > 4: mb.sigma_data.unsafe_free()
            if Int(mb.vndf_data) > 4: mb.vndf_data.unsafe_free()
            if Int(mb.vndf_marg) > 4: mb.vndf_marg.unsafe_free()
            if Int(mb.vndf_cond) > 4: mb.vndf_cond.unsafe_free()
            if Int(mb.lum_data) > 4: mb.lum_data.unsafe_free()
            if Int(mb.lum_marg) > 4: mb.lum_marg.unsafe_free()
            if Int(mb.lum_cond) > 4: mb.lum_cond.unsafe_free()
            if Int(mb.spectra_data) > 4: mb.spectra_data.unsafe_free()
        psc[unsafe_offset=0].measured_brdfs.unsafe_free()
    if psc[unsafe_offset=0].grid_count > 0:
        for gi in range(Int(psc[unsafe_offset=0].grid_count)):
            psc[unsafe_offset=0].grids[unsafe_offset=gi].density.unsafe_free()
        psc[unsafe_offset=0].grids.unsafe_free()
    if psc[unsafe_offset=0].nvdb_grid_count > 0:
        for gi in range(Int(psc[unsafe_offset=0].nvdb_grid_count)):
            if Int(psc[unsafe_offset=0].nvdb_grids[unsafe_offset=gi].blob) > 4:
                psc[unsafe_offset=0].nvdb_grids[unsafe_offset=gi].blob.unsafe_free()
        psc[unsafe_offset=0].nvdb_grids.unsafe_free()
    if Int(psc[unsafe_offset=0].light_sampler.cdf) > 4:
        psc[unsafe_offset=0].light_sampler.cdf.unsafe_free()
    # blas_nodes_arr/blas_primids_arr/instances are always real allocations
    # (min size 1, see finalize_scene) regardless of blas_count/instance_count.
    for bi in range(Int(psc[unsafe_offset=0].blas_count)):
        psc[unsafe_offset=0].blas_nodes_arr[unsafe_offset=bi].unsafe_free()
        psc[unsafe_offset=0].blas_primids_arr[unsafe_offset=bi].unsafe_free()
    psc[unsafe_offset=0].blas_nodes_arr.unsafe_free()
    psc[unsafe_offset=0].blas_primids_arr.unsafe_free()
    psc[unsafe_offset=0].blas_node_counts.unsafe_free()
    psc[unsafe_offset=0].blas_primid_counts.unsafe_free()
    psc[unsafe_offset=0].instances.unsafe_free()
    psc[unsafe_offset=0].template_mesh_start.unsafe_free()
    psc[unsafe_offset=0].template_mesh_end.unsafe_free()
    psc.unsafe_free()

def mojo_apply_overrides(
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    spp_override: Int32,
    w_override: Int32,
    h_override: Int32,
    seed_override: Int64 = Int64(-1),
):
    # The RNG seed is taken from the clock (see finalize_scene), so two runs
    # of the same build never produce the same image. That is the right
    # default for interactive/progressive rendering, but it makes A/B
    # measurement impossible: on a scene with a heavy-tailed estimator -- an
    # SMS caustic, say -- the whole-image mean swings by ~20% run to run, so
    # any comparison of two builds is measuring noise unless the seed is
    # pinned. `--seed` pins it.
    if seed_override >= Int64(0):
        # Hashed (splitmix64), not used raw. Every consumer builds its PCG
        # state by XOR-ing this seed with small counters, all on shared
        # streams, so a small raw seed made the RNGs structurally related:
        # seeded VCM runs were not distributed like unseeded ones (exact
        # flat-mirror glint: 20.2 seeded vs 22.6 unseeded vs 22.15 exact).
        var z = UInt64(seed_override) + UInt64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        psc[unsafe_offset=0].rng_seed = z ^ (z >> 31)
    if spp_override > Int32(0):
        var spp = spp_override
        var log2_spp = Int32(0)
        var tmp = spp
        while tmp > Int32(1):
            tmp >>= 1
            log2_spp += 1
        var log4_spp = (log2_spp + Int32(1)) / Int32(2)
        var dim = max(psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h)
        var log2_dim = Int32(0)
        var tmp_dim = dim
        while tmp_dim > Int32(1):
            tmp_dim >>= 1
            log2_dim += 1
        psc[unsafe_offset=0].samples_per_pixel = spp
        psc[unsafe_offset=0].log2_spp          = log2_spp
        psc[unsafe_offset=0].n_base4_digits    = log2_dim + log4_spp

    if w_override > Int32(0) and h_override > Int32(0):
        psc[unsafe_offset=0].film_w = w_override
        psc[unsafe_offset=0].film_h = h_override
        var cts = unsafe_alloc[Float32](16)
        make_perspective_matrix(psc[unsafe_offset=0].camera_fov, Float32(0.01), cts)
        var frame = Float32(w_override) / Float32(h_override)
        var smin_x: Float32; var smax_x: Float32
        var smin_y: Float32; var smax_y: Float32
        if frame >= Float32(1):
            smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
        else:
            smin_x = Float32(-1); smax_x = Float32(1)
            smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame
        var str_mat = unsafe_alloc[Float32](16)
        make_screen_to_raster(w_override, h_override,
                              smin_x, smax_x, smin_y, smax_y, str_mat)
        var rts = unsafe_alloc[Float32](16)
        _ = matrix_invert(str_mat, rts)
        var cts_inv = unsafe_alloc[Float32](16)
        _ = matrix_invert(cts, cts_inv)
        matrix_multiply(cts_inv, rts, psc[unsafe_offset=0].raster_to_camera)
        cts.unsafe_free(); str_mat.unsafe_free(); rts.unsafe_free(); cts_inv.unsafe_free()

        var log2_spp = psc[unsafe_offset=0].log2_spp
        var log4_spp = (log2_spp + Int32(1)) / Int32(2)
        var dim = max(w_override, h_override)
        var log2_dim = Int32(0)
        var tmp_dim = dim
        while tmp_dim > Int32(1):
            tmp_dim >>= 1
            log2_dim += 1
        psc[unsafe_offset=0].n_base4_digits = log2_dim + log4_spp

def mojo_parsed_scene_descriptor(
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    spectral: SpectralHandle,
) -> Pointer[SceneDescriptor2_C, MutUntrackedOrigin]:
    var sd = unsafe_alloc[SceneDescriptor2_C](1)
    sd[unsafe_offset=0].bvh2Nodes        = psc[unsafe_offset=0].bvh_nodes_cpu
    sd[unsafe_offset=0].primIds          = psc[unsafe_offset=0].prim_ids_cpu
    sd[unsafe_offset=0].meshes           = psc[unsafe_offset=0].meshes
    sd[unsafe_offset=0].meshCount        = Int64(psc[unsafe_offset=0].mesh_count)
    sd[unsafe_offset=0].materials        = psc[unsafe_offset=0].materials
    sd[unsafe_offset=0].materialCount    = Int64(psc[unsafe_offset=0].material_count)
    sd[unsafe_offset=0].areaLights       = psc[unsafe_offset=0].area_lights
    sd[unsafe_offset=0].areaLightCount   = Int64(psc[unsafe_offset=0].area_light_count)
    sd[unsafe_offset=0].textures         = psc[unsafe_offset=0].tex_filenames
    sd[unsafe_offset=0].textureCount     = Int64(psc[unsafe_offset=0].tex_count)
    sd[unsafe_offset=0].normalSlopeMaps  = psc[unsafe_offset=0].nmaps
    sd[unsafe_offset=0].distantLights    = psc[unsafe_offset=0].distant_lights
    sd[unsafe_offset=0].distantLightCount = Int64(psc[unsafe_offset=0].distant_count)
    sd[unsafe_offset=0].pointLights      = psc[unsafe_offset=0].point_lights
    sd[unsafe_offset=0].pointLightCount  = Int64(psc[unsafe_offset=0].point_count)
    sd[unsafe_offset=0].infiniteLights   = psc[unsafe_offset=0].infinite_lights
    sd[unsafe_offset=0].infiniteLightCount = Int64(psc[unsafe_offset=0].infinite_count)
    sd[unsafe_offset=0].spheres          = psc[unsafe_offset=0].spheres
    sd[unsafe_offset=0].sphereCount      = Int64(psc[unsafe_offset=0].sphere_count)
    sd[unsafe_offset=0].curves           = psc[unsafe_offset=0].curves
    sd[unsafe_offset=0].curveCount       = Int64(psc[unsafe_offset=0].curve_count)
    sd[unsafe_offset=0].mediums          = psc[unsafe_offset=0].mediums
    sd[unsafe_offset=0].mediumCount      = Int64(psc[unsafe_offset=0].medium_count)
    sd[unsafe_offset=0].mediumInterfaces = psc[unsafe_offset=0].medium_ifaces
    sd[unsafe_offset=0].mediumIfaceCount = Int64(psc[unsafe_offset=0].medium_iface_count)
    sd[unsafe_offset=0].grids            = psc[unsafe_offset=0].grids
    sd[unsafe_offset=0].gridCount        = Int64(psc[unsafe_offset=0].grid_count)
    sd[unsafe_offset=0].nvdbGrids        = psc[unsafe_offset=0].nvdb_grids
    sd[unsafe_offset=0].nvdbGridCount    = Int64(psc[unsafe_offset=0].nvdb_grid_count)
    sd[unsafe_offset=0].lightSampler    = psc[unsafe_offset=0].light_sampler
    sd[unsafe_offset=0].blasNodesArr    = psc[unsafe_offset=0].blas_nodes_arr
    sd[unsafe_offset=0].blasPrimIdsArr  = psc[unsafe_offset=0].blas_primids_arr
    sd[unsafe_offset=0].blasCount       = Int64(psc[unsafe_offset=0].blas_count)
    sd[unsafe_offset=0].instances       = psc[unsafe_offset=0].instances
    sd[unsafe_offset=0].instanceCount   = Int64(psc[unsafe_offset=0].instance_count)
    sd[unsafe_offset=0].measuredBrdfs      = psc[unsafe_offset=0].measured_brdfs
    sd[unsafe_offset=0].measuredBrdfCount  = Int64(psc[unsafe_offset=0].measured_count)
    sd[unsafe_offset=0].spectral        = spectral
    # CPU path never needs the GPU-resident texture array (shading.mojo's
    # _tex_lookup[False] branch uses sd.textures/textureCount above
    # instead) -- dangling/0, same convention every other GPU-only field
    # here would use if this were a GPU builder.
    sd[unsafe_offset=0].gpuTextures      = Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling()
    sd[unsafe_offset=0].gpuTextureCount  = Int64(0)
    return sd
