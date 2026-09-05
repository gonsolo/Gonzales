from std.math import sqrt, cos, sin, floor, acos, atan2, log2, exp, log, abs
from std.ffi import external_call
from std.memory import alloc
from .geometry import RGB, SampledSpectrum, Point3f, Point2f, Vec3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, MatKind, AreaLight_C, Sphere_C, Curve_C, CURVE_N_PIECES, curve_piece_endpoints, _curve_perp_axis, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, NormalSlopeMap_C, normal_slope_map_none, ShadowTask_C, LightSampler_C, light_sampler_sample, light_sampler_pdf, Instance_C, MeasuredBRDF_C, dot, cross, Frame, safe_sqrt, reflect, refract, schlick_fresnel, fr_dielectric, PI, TWO_PI, INV_PI, INV_FOUR_PI, _is_real_ptr, _atan2f
from .bxdf import BxDFSample, GeomContext, SobolSamples8, BxDFFlags, bxdf_is_delta, bxdf_sample_conductor, bxdf_sample_coated_conductor, bxdf_sample_dielectric, bxdf_sample_thin_dielectric, bxdf_eval_diffuse, bxdf_pdf_diffuse, bxdf_sample_diffuse, bxdf_sample_diffuse_transmit, ggx_D, ggx_G1, ggx_G2, ggx_vndf_pdf, bxdf_eval_conductor_ggx, bxdf_pdf_conductor_ggx, _nee_weight_simple, _nee_weight_hair, _nee_weight_simple_via_spectral, _nee_weight_coated_coat_lobe, _nee_weight_coated_diffuse_base
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_sample_measured, bxdf_pdf_measured, _nee_weight_measured
from .rng import PCG32
from .bvh import BVH2Node, SceneDescriptor2_C, any_hit_bvh2_core, ray_sphere_hit, traverse_bvh2_core, HairLobeConstants, _hair_precompute, _hair_eval_lobes, _hair_sample_dir, curve_offset_eps, LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee, _sample_infinite_light_textured, _equal_area_square_to_sphere, _equal_area_sphere_to_square
from .sampling import power_heuristic, sample_cosine_hemisphere, sample_cosine_hemisphere_world, sample_ggx_vndf, sobol_sample, mix_bits_u64
from .transform import transform_normal_by_instance
from .guide import GuideGrid, guide_pos_to_cell, guide_pdf, guide_sample, guide_cell_has_data, guide_record, null_guide, guide_is_active
from .spectrum import SpectralHandle, null_spectral_handle
from .reservoir import ReservoirState, reservoir_update, reservoir_finalize, reservoir_combine, reservoir_cap_confidence
from .restir_di import DIReservoir, di_reservoir_init, di_target_pdf, ReservoirIO, reservoir_io_null
from .restir_gi import GIReservoir, gi_reservoir_init, gi_target_pdf, GIReservoirIO, gi_reservoir_io_null, gi_temporal_spatial_combine
from .sms import (
    MAX_SMS_VERTICES, SMSVertex, sms_vertex_init, sms_vertex_flat, sms_vertex_sphere, sms_vertex_mats,
    mat22_mul, mat22_mul_v, mat22_inv, sms_solve_bernoulli, sms_walk, SMS_SOLVER_THRESHOLD,
    mnee_orthonormal_basis,
    sms_refresh_solved_frames,
)
from .restir_sms import SMSReservoir, sms_reservoir_init, sms_target_pdf, SMSReservoirIO, sms_reservoir_io_null

@fieldwise_init
struct GIPendingX1(TrivialRegisterPassable):
    """Phase 4's per-path-slot scratch carrying x1's own shading data forward
    from bounce 0 to bounce 1, so bounce 1 (if it also lands on a diffuse
    vertex) can weight its fresh GIReservoir candidate via gi_target_pdf --
    which needs x1's hit_point/normal/alb, no longer in scope by the time
    bounce 1's own _shade_diffuse_nee call runs. `active` distinguishes "no
    GI candidate pending" (0, the common case: bounce 0 wasn't diffuse, or
    ctx.use_restir is off) from "pending, awaiting bounce 1" (1) -- cleared
    back to 0 once bounce 1 consumes it (or left at 1 forever, harmlessly,
    if the path terminates/misses/hits non-diffuse before bounce 1, since
    this buffer's lifetime is exactly one render_tile call and is never
    reused across samples -- see rendering.mojo's allocation site).

    `throughput` MUST be captured here (path_ptr[].throughput at bounce 0,
    BEFORE shade_diffuse's own continuation-sampling epilogue multiplies it
    by x1's OWN real BSDF sample) -- reading path_ptr[].throughput fresh at
    bounce 1 instead would pick up x1's ORIGINAL sampled direction's
    f/cos/pdf factors, which have nothing to do with the RESAMPLED
    reconnection direction gi_resolve actually connects along. This was a
    real, shipped bug (~10x energy inflation, caught only once interactive
    mode's real per-bounce throughput evolution was exercised -- the
    earlier end-to-end unit test used a synthetic throughput=1 and never
    called the real shade_diffuse epilogue between bounces, so it couldn't
    have caught this)."""
    var active:     Int8
    var hit_point:  Vec3f
    var normal:     Vec3f
    var alb:        RGB
    var throughput: RGB

@always_inline
def gi_pending_x1_init() -> GIPendingX1:
    return GIPendingX1(
        active=Int8(0),
        hit_point=Vec3f(Float32(0)),
        normal=Vec3f(Float32(0)),
        alb=RGB(Float32(0)),
        throughput=RGB(Float32(0)),
    )

@fieldwise_init
struct LightContext(Copyable, Movable):
    """Light source arrays + counts + sampler, used by NEE. Split out of
    ShadeContext (step 7) so the grouping of "which lights exist in the scene"
    is self-contained instead of interleaved with scene/texture/sampling
    fields — construction call sites build this with keyword args precisely
    because several fields share the same Int/pointer type and a positional
    transposition wouldn't be caught by the type checker."""
    var area_lights:      UnsafePointer[AreaLight_C, MutExternalOrigin]
    var area_light_count: Int
    var distant_lights:   UnsafePointer[DistantLight_C, MutExternalOrigin]
    var distant_count:    Int
    var point_lights:     UnsafePointer[PointLight_C, MutExternalOrigin]
    var point_count:      Int
    var infinite_lights:  UnsafePointer[InfiniteLight_C, MutExternalOrigin]
    var infinite_count:   Int
    var spheres:          UnsafePointer[Sphere_C, MutExternalOrigin]
    var sphere_count:     Int
    var light_sampler:    LightSampler_C

@fieldwise_init
struct ShadeContext:
    """Bundles scene/texture/sampling data pointers passed to shade_nee_core
    and material shaders. Light-source data lives in the nested `lights`
    LightContext (step 7)."""
    var path_idx:         Int
    var bvh2Nodes:        UnsafePointer[BVH2Node, MutExternalOrigin]
    var primIds:          UnsafePointer[PrimId_C, MutExternalOrigin]
    var meshes:           UnsafePointer[TriangleMesh_C, MutExternalOrigin]
    var curves:           UnsafePointer[Curve_C, MutExternalOrigin]
    var materials:        UnsafePointer[Material_C, MutExternalOrigin]
    var tex_filenames:    UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin]
    var textures:         UnsafePointer[GpuTexture_C, MutExternalOrigin]
    var n_textures:       Int
    # Slope-space copies of the normal maps, indexed exactly like
    # `tex_filenames`/`textures` (so a Material_C's `normal_tex_idx`
    # addresses all three). Read ONLY by the SMS/MNEE manifold walk, which
    # needs the normal's derivatives and not just the normal -- see
    # geometry.mojo's NormalSlopeMap_C. Dangling on the GPU call sites (the
    # device-side scene upload has no slope maps yet, so a normal-mapped
    # caustic caster falls back to its smooth surface there, exactly as it
    # did before these existed).
    var nmaps:            UnsafePointer[NormalSlopeMap_C, MutExternalOrigin]
    var shadow_tasks:     UnsafePointer[ShadowTask_C, MutExternalOrigin]
    var px_scale:         Float32
    var sobol_matrices:   UnsafePointer[UInt32, MutExternalOrigin]
    var guide:            GuideGrid
    # Phase 2 (docs/A2_restir_migration_plan.md): CPU-only today, like
    # `guide` above -- every GPU ShadeContext construction site passes
    # False. When True, the diffuse material's area-light NEE (bounce 0
    # only) is replaced by RIS candidate generation + a single deferred
    # shadow-ray resolve (restir_di.mojo) instead of one direct NEE sample.
    var use_restir:       Bool
    var lights:           LightContext
    # Object instancing (see geometry.mojo's Instance_C docs). GPU call sites
    # pass dangling/zero-count values — GPU's own device-side scene upload
    # never populates instance data, so no PrimId_C.type==6 leaf can appear
    # there and these are never dereferenced on that path.
    var blasNodesArr:   UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin]
    var blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin]
    var instances:      UnsafePointer[Instance_C, MutExternalOrigin]
    # Staged spectral rendering rollout (Stage 2c, see project_spectral_rendering
    # memory) — host pointers on CPU, device pointers on GPU (see gpu.mojo's
    # ShadeContext construction sites).
    var spectral:       SpectralHandle
    # "measured" materials: one MeasuredBRDF_C per distinct .bsdf file,
    # indexed by Material_C.measured_idx (see measured_bsdf.mojo's loader).
    # Host pointer on CPU; dangling on GPU call sites until Stage 3 uploads
    # these as device buffers.
    var measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutExternalOrigin]
    # Phase 4 (docs/A2_restir_migration_plan.md), CPU-only, dead/unwired
    # (dangling/null on every call site except shade_core_cpu_nee's, which
    # is itself not called with real buffers by any render path yet -- see
    # render_tile/pipeline.mojo). `gi_pending` is per-PATH-SLOT scratch
    # (indexed by ctx.path_idx, lifetime = one render_tile call): whether
    # bounce 0 landed on a diffuse x1 with restir active, plus x1's own
    # hit_point/normal/alb needed to weight the fresh candidate once bounce
    # 1 resolves it (see GIPendingX1's docstring). `gi_io` is the frame-
    # wide, caller-owned read/write reservoir buffers + shared G-buffer
    # (same shape as restir_di.mojo's ReservoirIO, reused as-is for spatial
    # rejection since G-buffer data is a property of the pixel's primary
    # hit, not of which reservoir type is being reused) that
    # gi_temporal_spatial_combine/gi_resolve consume once bounce 1 lands on
    # a diffuse x2 too (this scope's restriction, see
    # _gi_generate_recon_candidate's docstring for why).
    var gi_pending: UnsafePointer[GIPendingX1, MutExternalOrigin]
    var gi_io:      GIReservoirIO

@always_inline
def _shading_normal(
    mesh: TriangleMesh_C,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    geo_normal: Vec3f,
    instance_idx: Int32 = Int32(-1),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
) -> Vec3f:
    """Interpolate per-vertex shading normals with barycentric (bu, bv).
    Falls back to the geometric normal if the mesh has no shading normals.
    The result is aligned to the same hemisphere as geo_normal, which the
    caller has already (a) oriented against the incoming ray and (b), if this
    hit came from an instanced BLAS (instance_idx >= 0), transformed to world
    space — mesh.normals are per-vertex OBJECT-space data in that case, so
    the interpolated result is transformed the same way before comparison."""
    if Int(mesh.normals) <= 4:
        return geo_normal
    var w0 = Float32(1.0) - bu - bv
    var n0 = Vec3f(mesh.normals[v0*3], mesh.normals[v0*3+1], mesh.normals[v0*3+2])
    var n1 = Vec3f(mesh.normals[v1*3], mesh.normals[v1*3+1], mesh.normals[v1*3+2])
    var n2 = Vec3f(mesh.normals[v2*3], mesh.normals[v2*3+1], mesh.normals[v2*3+2])
    var sn = n0 * w0 + n1 * bu + n2 * bv
    if instance_idx >= Int32(0):
        sn = transform_normal_by_instance(instances[Int(instance_idx)].worldToObj, sn)
    var slen = dot(sn, sn)
    if slen <= Float32(1e-12):
        return geo_normal
    sn = sn * (Float32(1.0) / sqrt(slen))
    if dot(sn, geo_normal) < Float32(0.0):
        sn = -sn
    return sn

@always_inline
def _srgb_to_linear(c: Float32) -> Float32:
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    else:
        return Float32(((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))

@always_inline
# Bilinear sample of ONE mip level: `off` = float offset of the level in
# tex.data, (lw, lh) = that level's dimensions. Pixel centres at +0.5, wrap.
@always_inline
def _sample_level(data: UnsafePointer[Float32, MutExternalOrigin], off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
    var s = u - Float32(Int(u))
    if s < Float32(0.0): s += Float32(1.0)
    var t = v - Float32(Int(v))
    if t < Float32(0.0): t += Float32(1.0)
    var fx = s * Float32(lw) - Float32(0.5)
    var fy = t * Float32(lh) - Float32(0.5)
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var wx = fx - Float32(x0); var wy = fy - Float32(y0)
    var x0w = ((x0 % lw) + lw) % lw
    var y0w = ((y0 % lh) + lh) % lh
    var x1w = (x0w + 1) % lw
    var y1w = (y0w + 1) % lh
    var i00 = off + (y0w * lw + x0w) * 3
    var i10 = off + (y0w * lw + x1w) * 3
    var i01 = off + (y1w * lw + x0w) * 3
    var i11 = off + (y1w * lw + x1w) * 3
    var w00 = (Float32(1.0) - wx) * (Float32(1.0) - wy)
    var w10 = wx * (Float32(1.0) - wy)
    var w01 = (Float32(1.0) - wx) * wy
    var w11 = wx * wy
    return RGB(
        data[i00]   * w00 + data[i10]   * w10 + data[i01]   * w01 + data[i11]   * w11,
        data[i00+1] * w00 + data[i10+1] * w10 + data[i01+1] * w01 + data[i11+1] * w11,
        data[i00+2] * w00 + data[i10+2] * w10 + data[i01+2] * w01 + data[i11+2] * w11,
    )

# Trilinear mip sample. lod 0 = base level (full res); higher = coarser.
# With a 1-level texture (no pyramid) this is plain bilinear on the base.
@always_inline
def _sample_tex(tex: GpuTexture_C, u: Float32, v: Float32, lod: Float32 = Float32(0.0)) -> RGB:
    var nl = Int(tex.n_levels)
    if nl <= 1:
        return _sample_level(tex.data, 0, Int(tex.width), Int(tex.height), u, v)
    var clamped = lod
    if clamped < Float32(0.0): clamped = Float32(0.0)
    var maxl = Float32(nl - 1)
    if clamped > maxl: clamped = maxl
    var l0 = Int(floor(clamped))
    var f = clamped - Float32(l0)
    # Walk to level l0, tracking its float offset and dims.
    var off = 0; var w = Int(tex.width); var h = Int(tex.height)
    for _k in range(l0):
        off += w * h * 3
        w = max(1, w // 2); h = max(1, h // 2)
    var c0 = _sample_level(tex.data, off, w, h, u, v)
    if f <= Float32(0.0) or l0 >= nl - 1:
        return c0
    var off1 = off + w * h * 3
    var w1 = max(1, w // 2); var h1 = max(1, h // 2)
    var c1 = _sample_level(tex.data, off1, w1, h1, u, v)
    return c0 + (c1 - c0) * f

# Unified 2D-texture fetch — the single use_gpu seam for texture sampling.
# GPU reads the uploaded GpuTexture_C table; CPU reads via OIIO by filename.
# (u, v) are the interpolated, NOT-yet-V-flipped coords; this applies pbrt's
# V-flip (1 - v) and (CPU) wrap. raw=True skips the sRGB decode (normal maps).
# Sets `found` False when there is no texture/data (caller uses its fallback).
@always_inline
def sample_texture[use_gpu: Bool](
    tex_idx: Int,
    u: Float32, v: Float32,
    raw: Bool,
    pixel_uv: Float32,   # texture-space footprint of one pixel (uv units); <=0 => LOD 0
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures: Int,
    mut found: Bool,
) -> RGB:
    found = False
    if tex_idx < 0:
        return RGB(Float32(0.0))
    var su = u
    var tv = Float32(1.0) - v  # pbrt V-flip: V=0 at top
    comptime if use_gpu:
        if tex_idx < n_textures:
            var tex = textures[tex_idx]
            if Int(tex.width) > 0:
                found = True
                # LOD = log2(texels covered by one pixel). pixel_uv is the uv
                # footprint; * width converts to texels. (CPU branch lets OIIO filter.)
                var texels = pixel_uv * Float32(tex.width)
                var lod = Float32(0.0)
                if texels > Float32(1.0):
                    lod = log2(texels)
                return _sample_tex(tex, su, tv, lod)
    else:
        if Int(tex_filenames) > 1:
            var filename = tex_filenames[tex_idx]
            if Int(filename) > 1:
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = alloc[Float32](3)
                tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    UnsafePointer[UInt8, MutExternalOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutExternalOrigin]](filename, su, tv, tr)
                var rr = tr[0]; var gg = tr[1]; var bb = tr[2]
                tr.free()
                found = True
                if raw:
                    return RGB(rr, gg, bb)
                return RGB(_srgb_to_linear(rr), _srgb_to_linear(gg), _srgb_to_linear(bb))
    return RGB(Float32(0.0))

@always_inline
def _tex_lookup[use_gpu: Bool](
    mat: Material_C,
    inter: Intersection_C,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh_C,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures: Int,
    pixel_uv: Float32 = Float32(0.0),
) -> RGB:
    var ti = Int(mat.tex_idx)
    if ti == -2:
        # Procedural checkerboard (pbrt "checkerboard" texture class): evaluated
        # analytically per-shading-point from Material_C's embedded checker_*
        # fields, identically on CPU and GPU — no image data or texture-table
        # index involved. See material_builder.mojo's "reflectance" handler.
        if Int(mesh.uvs) > 1:
            var w0 = Float32(1.0) - inter.u - inter.v
            var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
            var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
            # No V-flip here (unlike the image-texture path): pbrt evaluates its
            # checkerboard directly on the raw surface (u, v), with no notion of
            # image row order. Flipping v would toggle the sum-parity and swap
            # tex1/tex2 across the whole grid.
            var iu = Int(floor(su * mat.checker_uscale))
            var iv = Int(floor(tv * mat.checker_vscale))
            if (iu + iv) % 2 == 0:
                return mat.checker_tex1
            return mat.checker_tex2
        return mat.checker_tex1
    comptime if use_gpu:
        if ti >= 0 and ti < n_textures:
            var tex = textures[ti]
            if Int(tex.width) > 0:
                var w0 = Float32(1.0) - inter.u - inter.v
                var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
                var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
                tv = Float32(1.0) - tv  # PBRT V-flip: V=0 at top
                return _sample_tex(tex, su, tv)
    else:
        if ti >= 0 and Int(tex_filenames) > 8:
            var filename = tex_filenames[ti]
            if Int(filename) > 1 and Int(mesh.uvs) > 4:
                var w0 = Float32(1.0) - inter.u - inter.v
                var su = w0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
                var tv = w0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
                tv = Float32(1.0) - tv  # PBRT V-flip: V=0 at top
                su = su - Float32(Int(su))
                if su < Float32(0.0): su += Float32(1.0)
                tv = tv - Float32(Int(tv))
                if tv < Float32(0.0): tv += Float32(1.0)
                var tr = alloc[Float32](3)
                tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                _ = external_call["texture", Bool,
                    UnsafePointer[UInt8, MutExternalOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutExternalOrigin]](filename, su, tv, tr)
                var result = RGB(_srgb_to_linear(tr[0]), _srgb_to_linear(tr[1]), _srgb_to_linear(tr[2]))
                tr.free()
                return result
    return mat.albedo


@always_inline
def shade_core(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    tid: Int,
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return

    var inter = intersections[tid]
    if inter.hit == 0:
        path_ptr[].active = 0
        return

    var mat_idx = Int(inter.primId.materialIndex)
    var mat = materials[mat_idx]

    if mat.type == MatKind.area_light:
        path_ptr[].estimate += path_ptr[].throughput * mat.emission
        path_ptr[].active = 0
        return

    if mat.type == MatKind.diffuse:
        # Construct Normal
        var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
        if not ok:
            path_ptr[].active = 0
            return

        var p0 = Vec3f(mesh.points[v0 * 4], mesh.points[v0 * 4 + 1], mesh.points[v0 * 4 + 2])
        var p1 = Vec3f(mesh.points[v1 * 4], mesh.points[v1 * 4 + 1], mesh.points[v1 * 4 + 2])
        var p2 = Vec3f(mesh.points[v2 * 4], mesh.points[v2 * 4 + 1], mesh.points[v2 * 4 + 2])

        var (normal, ray_dir, _) = _geom_normal_and_ray(path_ptr, p0, p1, p2)

        # Cosine weighted hemisphere sampling
        var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state

        var r = sqrt(u1)
        var theta = 2.0 * PI * u2
        var x = r * cos(theta)
        var y = r * sin(theta)
        var z2 = 1.0 - u1
        var z = sqrt(z2 if z2 > 0.0 else Float32(0.0))

        # Build tangent basis
        var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
        var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
        var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > 0:
            dir = dir * (1.0 / sqrt(dlen))

        # Update Ray
        var org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) + ray_dir * inter.tHit + normal * 0.0001
        path_ptr[].ray = Ray_C(Point3f(org[0], org[1], org[2]), Vec3f(dir[0], dir[1], dir[2]))

        # Update Throughput (albedo)
        path_ptr[].throughput *= mat.albedo
    else:
        # Unknown material type — deactivate to prevent infinite loops
        path_ptr[].active = 0


# ── Triangle primitive helper ─────────────────────────────────────────────────
# Decodes the primId encoding into (mesh, v0, v1, v2). Returns ok=False for
# non-triangle hits (caller should deactivate path and return).
@always_inline
def _get_tri_verts(
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
) -> Tuple[TriangleMesh_C, Int, Int, Int, Bool]:
    var mi: Int
    var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1)
        bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32)
        bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        return (meshes[0], 0, 0, 0, False)
    var m = meshes[mi]
    return (m, Int(m.vertexIndices[bv]), Int(m.vertexIndices[bv+1]), Int(m.vertexIndices[bv+2]), True)


# ── Shading frame helper ──────────────────────────────────────────────────────
# Computes geom normal from triangle cross product, normalizes, faceforwards it
# toward the incoming ray, and extracts ray_dir/ray_org from path state.
# Returns (geom_normal_ff, ray_dir, ray_org).
@always_inline
def _geom_normal_and_ray(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    p0: Vec3f,
    p1: Vec3f,
    p2: Vec3f,
    instance_idx: Int32 = Int32(-1),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
) -> Tuple[Vec3f, Vec3f, Vec3f]:
    """p0/p1/p2 come straight from `mesh.points` — object-space if this hit
    came from an instanced BLAS (instance_idx >= 0), so the resulting normal
    is transformed to world space before face-forwarding against the ray
    (which is always world-space). The hit point (`ro`, the ray origin the
    caller combines with tHit) needs no such fixup — tHit is preserved 1:1
    between object/world parameterizations, see bvh.mojo's
    _transform_ray_to_instance_space."""
    var gn = cross(p1 - p0, p2 - p0)
    if instance_idx >= Int32(0):
        gn = transform_normal_by_instance(instances[Int(instance_idx)].worldToObj, gn)
    var nlen = dot(gn, gn)
    if nlen > Float32(0.0):
        gn = gn * (Float32(1.0) / sqrt(nlen))
    var rd = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(gn, rd) > Float32(0.0):
        gn = -gn
    var ro = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    return (gn, rd, ro)


# Analytic-sphere counterpart of _geom_normal_and_ray: exact outward normal
# at the hit point ((hit - center)/radius), face-forwarded toward the
# incoming ray. Spheres are never instanced (see Instance_C's docs), so
# unlike the triangle path there is no object/world transform to apply.
@always_inline
def _sphere_geom_normal_and_ray(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
) -> Tuple[Vec3f, Vec3f, Vec3f]:
    var sph = spheres[Int(inter.primId.id1)]
    var rd = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ro = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit = ro + rd * inter.tHit
    var c = Vec3f(sph.center.x, sph.center.y, sph.center.z)
    var gn = hit - c
    var nlen = dot(gn, gn)
    if nlen > Float32(0.0):
        gn = gn * (Float32(1.0) / sqrt(nlen))
    if dot(gn, rd) > Float32(0.0):
        gn = -gn
    return (gn, rd, ro)


# ── Unified per-hit geometry (triangle OR analytic sphere) ────────────────────
# Every material shader used to hand-roll its own "if primId.type==4: sphere
# branch, else: _get_tri_verts + cross product" — the same handful of lines
# copy-pasted into 8 different functions. This is the ONE place that decides
# how to turn an Intersection_C into (geo_normal, ray_dir, ray_org), for
# either primitive type; callers needing UV-dependent extras (textures,
# normal maps, anisotropy tangents, shading-normal interpolation — all of
# which have no sphere analogue, since Sphere_C has no UV parameterization)
# still branch on `is_sphere` themselves, but only for THAT material-specific
# logic, not for re-deriving the normal/ray every time.
# `mesh`/v0/v1/v2 are only meaningful when is_sphere is False; when ok is
# False the caller must deactivate the path without reading anything else.
@always_inline
def _hit_geom(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    instance_idx: Int32 = Int32(-1),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
) -> Tuple[Bool, Bool, Vec3f, Vec3f, Vec3f, TriangleMesh_C, Int, Int, Int]:
    """Returns (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2)."""
    if inter.primId.type == Int8(4):
        var sph_r = _sphere_geom_normal_and_ray(path_ptr, inter, spheres)
        return (True, True, sph_r[0], sph_r[1], sph_r[2], meshes[0], 0, 0, 0)
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        var z = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (False, False, z, z, z, mesh, 0, 0, 0)
    var p0 = Vec3f(mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = Vec3f(mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = Vec3f(mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
    var gnr = _geom_normal_and_ray(path_ptr, p0, p1, p2, instance_idx, instances)
    return (True, False, gnr[0], gnr[1], gnr[2], mesh, v0, v1, v2)


@always_inline
def _shadow_contribute[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    origin: Vec3f,
    dir: Vec3f,
    tmax: Float32,
    contrib: RGB,
    guide_write: GuideGrid = null_guide(),
):
    comptime if enqueue_shadow:
        ctx.shadow_tasks[ctx.path_idx] = ShadowTask_C(
            Point3f(origin[0], origin[1], origin[2]),
            Vec3f(dir[0], dir[1], dir[2]),
            tmax, RGB(contrib.r, contrib.g, contrib.b), Int32(1), Int32(0))
    else:
        var shadow_ray = Ray_C(Point3f(origin[0], origin[1], origin[2]), Vec3f(dir[0], dir[1], dir[2]))
        if not any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, shadow_ray, tmax,
                                  ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                                  ctx.lights.spheres, ctx.lights.sphere_count):
            path_ptr[].estimate += contrib
            # Record in the guide at the PARENT surface (one bounce back):
            # "scatter direction ray.dir from parent_cell leads to illumination W here."
            # This teaches indirect-illumination guiding — path_ptr[].ray.origin is where
            # the path came from and ray.direction is the scatter direction that arrived here.
            if guide_is_active(guide_write) and path_ptr[].bounce > 0:
                var parent_cell = guide_pos_to_cell(guide_write, path_ptr[].ray.origin)
                if parent_cell >= 0:
                    var w = contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)
                    # Normalize by current throughput to record incoming radiance at
                    # the parent surface, independent of path history (Li, not T*Li).
                    var t = path_ptr[].throughput
                    var t_lum = t.r * Float32(0.2126) + t.g * Float32(0.7152) + t.b * Float32(0.0722)
                    if t_lum > Float32(1e-7):
                        w = w / t_lum
                    if w > Float32(1e-7):
                        var sd = path_ptr[].ray.direction
                        guide_record(guide_write, parent_cell, sd.x, sd.y, sd.z, w)


# ── DiffuseTransmission branch ────────────────────────────────────────────────
@always_inline
def shade_diffuse_transmission[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
):
    var mat = ctx.materials[Int(inter.primId.materialIndex)]
    var (ok, is_sphere, normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances)
    if not ok:
        path_ptr[].active = 0
        return

    # "texture reflectance"/"texture transmittance" (e.g. a leaf.tga imagemap)
    # both resolve to the same mat.tex_idx (Material_C has one texture slot,
    # shared across kinds) — sample it once and use it for both lobes. Leaving
    # this at the flat mat.albedo/mat.emission default (as before) rendered
    # textured diffusetransmission foliage as a dull, wall-coloured grey blob
    # instead of the actual leaf texture, effectively invisible against a
    # similarly-toned wall. No UV space exists on a sphere, so it always
    # keeps the flat default.
    var refl = mat.albedo
    var trans = mat.emission
    if not is_sphere and Int(mat.tex_idx) != -1:
        var tex_rgb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        refl = tex_rgb
        trans = tex_rgb

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, bounce_normal, lobe_alb, lobe_w, choose_reflect) = bxdf_sample_diffuse_transmit(
        normal, refl, trans, pcg.next_float(), pcg.next_float(), pcg.next_float())
    if bs.is_valid == Int8(0):
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    var hit_point = ray_org + ray_dir * inter.tHit + bounce_normal * Float32(0.0001)

    # ── NEE direct light sampling (MIS weighted, with MNEE glass caustics) ─────
    # Shares _nee_area_lights with plain diffuse — including its MNEE probe for
    # up to 2 glass surfaces between hit_point and the light — parameterized by
    # the chosen lobe's normal/albedo and its selection-weight compensation.
    _nee_area_lights[enqueue_shadow](path_ptr, ctx, bounce_normal, hit_point, lobe_alb,
        pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg, null_guide(), lobe_w)

    # ── Distant/point/sphere/infinite NEE, via the shared Light interface +
    # BxDF interface (_nee_weight_simple, mat_kind=0=diffuse — this lobe's
    # BRDF is Lambertian) ────────────────────────────────────────────────────
    # This material previously had NO direct lighting from these light types
    # at all (only area lights, above) — a severe gap for infinite-light-only
    # scenes with diffusetransmission-heavy foliage (e.g. sanmiguel-courtyard:
    # every transmissive leaf/vine surface relied purely on noisy BSDF-escape
    # rays to ever see the env map's sun, producing systematic under-lighting
    # — darker, blue-sky-dominated average — plus severe fireflies from the
    # rare lucky hits). lobe_w compensates for the stochastic reflect/
    # transmit lobe selection, same as _nee_area_lights above.
    var wo_dt = -ray_dir
    for dl_i in range(ctx.lights.distant_count):
        var ls_d = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i])
        var w_d = _nee_weight_simple_via_spectral(ls_d, Int32(0), lobe_alb, Float32(0), bounce_normal, wo_dt, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths) * lobe_w
        if not w_d.is_black():
            var contrib_d = path_ptr[].throughput * w_d
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_d.wi, ls_d.dist, contrib_d)
    for pl_i in range(ctx.lights.point_count):
        var ls_p = _sample_point_light_nee(ctx.lights.point_lights[pl_i], hit_point)
        var w_p = _nee_weight_simple_via_spectral(ls_p, Int32(0), lobe_alb, Float32(0), bounce_normal, wo_dt, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths) * lobe_w
        if not w_p.is_black():
            var contrib_p = path_ptr[].throughput * w_p
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_p.wi, ls_p.dist * Float32(0.9999), contrib_p)
    for sph_i in range(ctx.lights.sphere_count):
        var ls_sph = _sample_sphere_light_nee(ctx.lights.spheres[sph_i], ctx.lights.sphere_count, hit_point, pcg)
        var w_sph = _nee_weight_simple_via_spectral(ls_sph, Int32(0), lobe_alb, Float32(0), bounce_normal, wo_dt, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths) * lobe_w
        if not w_sph.is_black():
            var contrib_sph = path_ptr[].throughput * w_sph
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_sph.wi, ls_sph.dist * Float32(0.9999), contrib_sph)
    for inf_i in range(ctx.lights.infinite_count):
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var w_e = _nee_weight_simple_via_spectral(ls_e, Int32(0), lobe_alb, Float32(0), bounce_normal, wo_dt, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths) * lobe_w
        if not w_e.is_black():
            var contrib_e = path_ptr[].throughput * w_e
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_e.wi, ls_e.dist, contrib_e)

    # ── BSDF scatter ───────────────────────────────────────────────────────────
    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))

    # Throughput: lobe_alb / pdf_bsdf * lobe_selection_weight
    # = lobe_alb / (cos/π) * (total/p_lobe) → lobe_alb * π/cos * lobe_w
    # But cosine-hemisphere importance sampling gives cos/π cancel:
    # f * cos / pdf = (lobe_alb/π) * cos / (cos/π) = lobe_alb
    # Then multiply by lobe_w to compensate for stochastic lobe selection.
    path_ptr[].throughput *= lobe_alb * lobe_w

    # Store BSDF pdf for next-bounce MIS (cosine hemisphere: cos/π)
    path_ptr[].lastBsdfPdf = bs.pdf
    path_ptr[].specularBounce = Int8(0)

    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = refl
    path_ptr[].bounce += 1

    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, pcg, u_rr)


# ── CoatedDiffuse (plastic) branch ───────────────────────────────────────────
@always_inline
def _albedo_highlight_boost(albedo: RGB, contrib: RGB) -> RGB:
    """Nudge the albedo AOV toward white in proportion to a highlight-strength
    NEE contribution. Saturates via 1-exp(-k*luma) rather than linearly
    (min(1,luma)) so even a modest per-sample hit — rare across an spp
    average, since a sharp/sparse specular contribution looks identical to
    its dark neighbours in a single sample — visibly elevates the denoiser's
    guide buffer instead of needing many such hits to add up. Without a
    strong-enough boost here, the coat's specular highlights (and, since
    2026-07-08, the multi-tap base-recycling energy) carry correct energy in
    beauty but get smoothed away by the denoiser anyway because albedo never
    discriminates the highlight pixels from their neighbours strongly enough.
    """
    var luma = contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)
    var boost = Float32(1.0) - exp(-luma * Float32(4.0))
    return albedo + (RGB(Float32(1.0)) - albedo) * boost

@always_inline
def shade_coated_diffuse[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances)
    if not ok:
        path_ptr[].active = 0
        return

    var alb: RGB
    var normal: Vec3f
    if is_sphere:
        # Analytic sphere: exact normal IS the shading normal, no UV space
        # so alb falls back to the material's flat albedo.
        alb = mat.albedo
        normal = geo_normal
    else:
        alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        # Use interpolated shading normal (geometric normal still drives hit-point offset)
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal, inter.primId.instanceIdx, ctx.instances)

    var hit_point = ray_org + ray_dir * inter.tHit + geo_normal * Float32(0.0001)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = alb

    # ── Layered BSDF: smooth dielectric coat over a Lambertian base ──────────
    # Stochastic random walk (PBRT LayeredBxDF, CoatedDiffuse). The coat's air
    # interface reflects a glossy lobe (exact dielectric Fresnel) and transmits
    # the rest into the coat, where light scatters off the diffuse base and is
    # partially recycled by (total internal) reflection at the coat underside.
    # That multiple scattering is what brightens and saturates the base colour.
    # Fresnel at each interface is handled by the reflect/transmit probability
    # split, so the throughput accumulator beta only gathers the base albedo.
    var ior = mat.emission.r            # coat IOR (η_coat/η_air), set at parse
    var inv_ior = Float32(1.0) / ior
    # roughU/V already hold the resolved GGX alpha (see _psc_handle_make_named_material's
    # remaproughness handling). 0 ⇒ smooth mirror coat (e.g. car paint 0.001);
    # larger ⇒ soft sheen (e.g. tyres 0.4).
    var coat_alpha = max(mat.roughU, mat.roughV)
    var is_rough_coat = coat_alpha > Float32(0.001)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])  # toward viewer

    # Tangent frame (Frisvad) around the shading normal for hemisphere sampling.
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

    # Microfacet normal at the coat's air interface: the surface normal when
    # smooth, else a GGX visible-normal sample (Heitz VNDF). Fresnel is taken
    # at this microfacet; the VNDF G2/G1 weight is approximated as 1 (matching
    # the conductor path), so the throughput accumulator stays albedo-only.
    var wm = normal
    if is_rough_coat:
        var wo_l = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, normal))
        var wm_l = sample_ggx_vndf(wo_l, coat_alpha, coat_alpha, pcg.next_float(), pcg.next_float())
        wm = tangent * wm_l.x + bitangent * wm_l.y + normal * wm_l.z
        var wmlen = dot(wm, wm)
        if wmlen > Float32(0.0):
            wm = wm * (Float32(1.0) / sqrt(wmlen))
    var cos_wm = dot(wo, wm)
    var f_entry = fr_dielectric(cos_wm, ior)
    var cos_o = dot(wo, normal)

    # Rough coat: NEE against area lights and distant lights, MIS-combined
    # with the reflected ray below. Fired unconditionally (NOT gated on the
    # reflect-vs-transmit coin flip a few lines down) because it evaluates
    # the coat's own BRDF response — a surface property, independent of
    # which lobe this particular sample's *continuation* ray happens to
    # follow. Gating it on that coin flip would double-count the interface
    # Fresnel term (once implicitly via the gate's own probability, once via
    # the NEE half-vector's own Fresnel term computed below) and silently
    # bias the result low by roughly that probability — this was tried
    # first and measurably under-shot pbrt's reference brightness.
    # A glossy lobe's reflection cone is too narrow for naive BSDF sampling
    # to reliably find compact/distant lights (confirmed by a real missing
    # highlight — lamp's shade never picked up its bulb's reflection even at
    # 512spp without this). Smooth coat (is_rough_coat false, e.g. car
    # paint) skips NEE entirely: a delta reflection can never land on a
    # stochastically-sampled light direction, so any shadow ray fired there
    # would just be wasted work.
    if is_rough_coat and cos_o > Float32(0.0):
        # Coat's own glossy dielectric lobe against every light type, via the
        # shared Light interface (LightSample samplers, bvh.mojo) + the
        # coat-specific BxDF weight (_nee_weight_coated_coat_lobe, bxdf.mojo)
        # — one call per light instead of the same D*G2*F/(4*cos_o) math
        # hand-inlined once per light type. Previously only area+distant
        # were covered; sphere/point/infinite were silently missing.
        var ls_area_c = _sample_area_light_nee(ctx, hit_point, pcg)
        var w_area_c = _nee_weight_coated_coat_lobe(ls_area_c, ior, coat_alpha, normal, wo)
        if not w_area_c.is_black():
            var contrib_area_c = path_ptr[].throughput * w_area_c
            # See _albedo_highlight_boost: NEE shadow rays never otherwise
            # touch albedo, so a sharp specular highlight looks identical to
            # its dark neighbours in the denoiser's guide buffer and gets
            # smoothed away without this.
            path_ptr[].albedo = _albedo_highlight_boost(path_ptr[].albedo, contrib_area_c)
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_area_c.wi, ls_area_c.dist * Float32(0.9999), contrib_area_c)

        for sph_i_coat in range(ctx.lights.sphere_count):
            var ls_sph_coat = _sample_sphere_light_nee(ctx.lights.spheres[sph_i_coat], ctx.lights.sphere_count, hit_point, pcg)
            var w_sph_coat = _nee_weight_coated_coat_lobe(ls_sph_coat, ior, coat_alpha, normal, wo)
            if not w_sph_coat.is_black():
                var contrib_sph_coat = path_ptr[].throughput * w_sph_coat
                path_ptr[].albedo = _albedo_highlight_boost(path_ptr[].albedo, contrib_sph_coat)
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_sph_coat.wi, ls_sph_coat.dist * Float32(0.9999), contrib_sph_coat)

        for dl_i_coat in range(ctx.lights.distant_count):
            var ls_dl_coat = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i_coat])
            var w_dl_coat = _nee_weight_coated_coat_lobe(ls_dl_coat, ior, coat_alpha, normal, wo)
            if not w_dl_coat.is_black():
                var contrib_dl_coat = path_ptr[].throughput * w_dl_coat
                path_ptr[].albedo = _albedo_highlight_boost(path_ptr[].albedo, contrib_dl_coat)
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_dl_coat.wi, ls_dl_coat.dist, contrib_dl_coat)

        for pl_i_coat in range(ctx.lights.point_count):
            var ls_pl_coat = _sample_point_light_nee(ctx.lights.point_lights[pl_i_coat], hit_point)
            var w_pl_coat = _nee_weight_coated_coat_lobe(ls_pl_coat, ior, coat_alpha, normal, wo)
            if not w_pl_coat.is_black():
                var contrib_pl_coat = path_ptr[].throughput * w_pl_coat
                path_ptr[].albedo = _albedo_highlight_boost(path_ptr[].albedo, contrib_pl_coat)
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_pl_coat.wi, ls_pl_coat.dist * Float32(0.9999), contrib_pl_coat)

        for inf_i_coat in range(ctx.lights.infinite_count):
            var ls_inf_coat = _sample_infinite_light_nee(ctx.lights.infinite_lights[inf_i_coat], Point2f(pcg.next_float(), pcg.next_float()))
            var w_inf_coat = _nee_weight_coated_coat_lobe(ls_inf_coat, ior, coat_alpha, normal, wo)
            if not w_inf_coat.is_black():
                var contrib_inf_coat = path_ptr[].throughput * w_inf_coat
                path_ptr[].albedo = _albedo_highlight_boost(path_ptr[].albedo, contrib_inf_coat)
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_inf_coat.wi, ls_inf_coat.dist, contrib_inf_coat)

    if pcg.next_float() < f_entry:
        # Glossy reflection off the coat (rough ⇒ GGX lobe, smooth ⇒ mirror).
        var refl = wm * (Float32(2.0) * cos_wm) - wo
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        if dot(refl, normal) <= Float32(0.0):
            path_ptr[].active = 0          # reflected below the surface — discard
            path_ptr[].pcgState = pcg.state
            return

        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(refl[0], refl[1], refl[2]))
        if is_rough_coat:
            # MIS-gate the reflected ray against the NEE above (real pdf_bsdf,
            # specularBounce=0) instead of the delta-lobe full-credit path.
            var d_sampled = ggx_D(dot(normal, wm), coat_alpha)
            path_ptr[].specularBounce = Int8(0)
            path_ptr[].lastBsdfPdf = ggx_vndf_pdf(cos_o, cos_wm, d_sampled, coat_alpha)
        else:
            # Smooth mirror coat: single-strategy lobe (no NEE) ⇒
            # specularBounce=1 takes full light on miss/hit, pdf is irrelevant.
            path_ptr[].specularBounce = Int8(1)
            path_ptr[].lastBsdfPdf = Float32(0.0)
        path_ptr[].bounce += 1
        path_ptr[].pcgState = pcg.state
        return

    # Transmitted into the coat: random-walk the base/coat-underside layers.
    var beta = RGB(Float32(1.0))
    var exited = False
    var exit_dir = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))

    comptime MAX_COAT_DEPTH = 10
    for depth in range(MAX_COAT_DEPTH):
        # Russian-roulette the recycling walk itself once `beta` (the base
        # albedo raised to the number of prior internal-reflection bounces at
        # THIS hit point) has decayed enough that further bounces contribute
        # negligibly — mirrors PBRT LayeredBxDF::f()'s own `depth>3` RR gate.
        # Needed because NEE now fires every iteration (see below), not just
        # the first, so an unbounded walk would mean unbounded shadow rays.
        if depth > 3:
            var beta_max = max(beta.r, max(beta.g, beta.b))
            if beta_max < Float32(0.25):
                var q_rr = max(Float32(0.0), Float32(1.0) - beta_max)
                if pcg.next_float() < q_rr:
                    break
                beta = beta * (Float32(1.0) / (Float32(1.0) - q_rr))

        # ── Diffuse base: NEE at THIS bounce, weighted by the coat's
        #    transmittance for the incoming light direction AND `beta` — the
        #    accumulated base-albedo attenuation from any prior recycled
        #    bounces at this same hit point (still 1.0 on the first pass, so
        #    this exactly reproduces the old single-scatter formula there).
        #    Firing every iteration instead of only the first is what turns
        #    this into "full stochastic NEE": PBRT's LayeredBxDF::f() gets the
        #    same TRT/TRTRT/... multi-scatter terms from ONE correlated random
        #    walk reusing a single virtual-light sample; gonzales instead
        #    draws an independent fresh light sample per bounce (decorrelated,
        #    same expected energy, simpler to reason about). The RR gate above
        #    keeps this from growing shadow-ray traffic unboundedly on
        #    high-albedo/grazing-angle coats that recycle many times. ──
        # Diffuse base against every light type via the shared Light
        # interface + the coat-transmittance-aware BxDF weight
        # (_nee_weight_coated_diffuse_base, bxdf.mojo) — `beta` (this
        # bounce's accumulated recycled-albedo attenuation) is applied by
        # the caller here, not inside the weight function, since it's walk
        # state the interface's flat per-LightSample signature can't hold.
        # Previously only area+distant were covered; sphere was entirely
        # missing (see the pbrt-book bug this closed: a scene lit ONLY by
        # sphere lights rendered this material almost completely black) and
        # point was entirely missing too. Infinite lights (below, unchanged)
        # deliberately keep their own textured-CDF/cosine-hemisphere-fallback
        # sampling rather than routing through the generic uniform-sphere
        # fallback sampler — same rationale as diffuse's own infinite-light
        # NEE (see project_light_bxdf_interfaces memory).
        var ls_area = _sample_area_light_nee(ctx, hit_point, pcg)
        var w_area = _nee_weight_coated_diffuse_base(ls_area, alb, ior, normal)
        if not w_area.is_black():
            var contrib_area = path_ptr[].throughput * beta * w_area
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_area.wi, ls_area.dist * Float32(0.9999), contrib_area)

        for sph_i in range(ctx.lights.sphere_count):
            var ls_sph = _sample_sphere_light_nee(ctx.lights.spheres[sph_i], ctx.lights.sphere_count, hit_point, pcg)
            var w_sph = _nee_weight_coated_diffuse_base(ls_sph, alb, ior, normal)
            if not w_sph.is_black():
                var contrib_sph = path_ptr[].throughput * beta * w_sph
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_sph.wi, ls_sph.dist * Float32(0.9999), contrib_sph)

        for pl_i in range(ctx.lights.point_count):
            var ls_pl = _sample_point_light_nee(ctx.lights.point_lights[pl_i], hit_point)
            var w_pl = _nee_weight_coated_diffuse_base(ls_pl, alb, ior, normal)
            if not w_pl.is_black():
                var contrib_pl = path_ptr[].throughput * beta * w_pl
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_pl.wi, ls_pl.dist * Float32(0.9999), contrib_pl)

        # ── Env-map (infinite light) NEE at the base, every bounce (see area
        #    lights above). Light enters via the coat so the contribution is
        #    weighted by 1 - F(cos_env). The view-side coat transmittance is
        #    implicit in reaching this branch (and, for depth>0, in `beta`). ──
        if ctx.lights.infinite_count > 0:
            for inf_i in range(ctx.lights.infinite_count):
                var ilight = ctx.lights.infinite_lights[inf_i]
                var env_dir: Vec3f
                var env_rgb: RGB
                var pdf_light: Float32
                if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and _is_real_ptr(ilight.cdf_ptr) and ilight.cdf_w > Int32(0):
                    var u1_env = pcg.next_float()
                    var u2_env = pcg.next_float()
                    var (dir_v, rgb_v, pdf_v) = _sample_infinite_light_textured(ilight, Point2f(u1_env, u2_env))
                    env_dir = dir_v.to_simd()
                    env_rgb = rgb_v
                    pdf_light = pdf_v
                else:
                    var _env_s = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), normal)
                    env_dir = _env_s[0]
                    pdf_light = _env_s[1]
                    env_rgb = ilight.scale
                var cos_env = dot(normal, env_dir)
                if cos_env > Float32(0.0) and not env_rgb.is_black() and pdf_light > Float32(0.0):
                    var t_env = Float32(1.0) - fr_dielectric(cos_env, ior)
                    var pdf_bsdf_nee = cos_env / PI
                    var mis_w = power_heuristic(pdf_light, pdf_bsdf_nee)
                    var contrib_e = path_ptr[].throughput * beta * alb * env_rgb * (cos_env * t_env / (PI * pdf_light)) * mis_w
                    var t_max_env = Float32(100000.0)
                    _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, env_dir, t_max_env, contrib_e)

        # Distant light NEE through the coat (delta light: MIS weight = 1),
        # every bounce (see area lights above) — now correctly `beta`-weighted
        # per depth instead of the old fire-once gate, so no double-counting.
        for dl_i in range(ctx.lights.distant_count):
            var ls_dl = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i])
            var w_dl = _nee_weight_coated_diffuse_base(ls_dl, alb, ior, normal)
            if not w_dl.is_black():
                var contrib_dl = path_ptr[].throughput * beta * w_dl
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_dl.wi, ls_dl.dist, contrib_dl)

        # Lambertian base: sample a cosine-weighted up-going direction.
        var _w_up_sample = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), normal)
        var w_up = _w_up_sample[0]
        beta *= alb
        # Chrominance floor: up to MAX_COAT_DEPTH iterations of `beta *= alb`
        # against the SAME cached texture sample (one _tex_lookup per shading
        # call, reused for the whole recycling walk -- see `alb` above) means
        # beta = base_albedo^depth. That's the intended "saturation" effect
        # (see this function's own docstring), but for a texture whose
        # specific texel has even a modest per-channel imbalance (ordinary
        # texture variation -- a wood-grain fleck, a tile-grout pixel),
        # raising it to the 10th power drives the weakest channel toward
        # zero while another stays large -- a real, visible magenta/green-
        # starved artifact found by comparing against the scene's published
        # reference image (a thin ceiling/glass-edge highlight was purple
        # instead of the reference's green). Same class of bug as the
        # RR-throughput and spectral-NEE fixes earlier this session:
        # legitimate per-step math, unboundedly extreme after enough
        # repetitions. Floor the weakest channel at a small fraction of the
        # strongest each iteration to bound how extreme a single texel's
        # saturation can compound to, while still letting real,
        # order-of-magnitude color saturation through.
        var beta_max_c = max(beta.r, max(beta.g, beta.b))
        comptime BETA_CHROMA_FLOOR: Float32 = 0.1
        var beta_floor = beta_max_c * BETA_CHROMA_FLOOR
        if beta.r < beta_floor: beta.r = beta_floor
        if beta.g < beta_floor: beta.g = beta_floor
        if beta.b < beta_floor: beta.b = beta_floor

        # Coat underside: transmit out (exit) or reflect back (recycle).
        # The interface microfacet is the surface normal when smooth, else a
        # GGX VNDF sample seen from w_up — this softens the exit direction.
        var wm_e = normal
        if is_rough_coat:
            var wup_l = Vec3f(dot(w_up, tangent), dot(w_up, bitangent), dot(w_up, normal))
            var wm_e_l = sample_ggx_vndf(wup_l, coat_alpha, coat_alpha, pcg.next_float(), pcg.next_float())
            wm_e = tangent * wm_e_l.x + bitangent * wm_e_l.y + normal * wm_e_l.z
            var wmelen = dot(wm_e, wm_e)
            if wmelen > Float32(0.0):
                wm_e = wm_e * (Float32(1.0) / sqrt(wmelen))
        var cos_up = dot(w_up, wm_e)
        var f_exit = fr_dielectric(cos_up, inv_ior)
        if pcg.next_float() < (Float32(1.0) - f_exit):
            var rr = refract(Vec3f(-w_up[0], -w_up[1], -w_up[2]),
                             Vec3f(-wm_e[0], -wm_e[1], -wm_e[2]), ior)
            if rr[0]:
                var wt = rr[1]
                exit_dir = Vec3f(wt.x, wt.y, wt.z)
                var elen = dot(exit_dir, exit_dir)
                if elen > Float32(0.0):
                    exit_dir = exit_dir * (Float32(1.0) / sqrt(elen))
                # A rough microfacet can refract below the surface; recycle then.
                if dot(exit_dir, normal) > Float32(0.0):
                    exited = True
                    break
            # Refraction failed / exited below surface → fall through and recycle.
        # Internal reflection: light recycled; next iteration re-samples base.

    if not exited:
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(exit_dir[0], exit_dir[1], exit_dir[2]))
    # NEE-only for direct lighting: the layered exit ray's true pdf is
    # intractable (refracted, multi-bounce) so it can't be MIS-combined with
    # NEE. Setting lastBsdfPdf = 0 makes the miss/emitter handlers drop this
    # ray's *direct* light contribution (mis_weight → 0); NEE supplies direct
    # lighting while the exit ray still carries indirect bounces. Avoids the
    # double-count between the multi-scatter exit and single-scatter NEE.
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].throughput *= beta
    path_ptr[].bounce += 1

    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, pcg, u_rr)


# Cap on throughput luminance immediately after an RR survival compensation
# (see _apply_russian_roulette below). Each individual RR event is still the
# standard unbiased estimator (throughput /= survival_probability) -- the
# problem this guards against is a RARE STREAK of survivals compounding
# across many bounces, seen concretely on a path trapped doing many
# consecutive total-internal-reflection bounces inside glass (each
# survival's ~2x compensation is unremarkable alone, but 2^30 is not).
# Clamping the RESULT after each application trades a small, deliberate
# bias for a large variance reduction -- the same trade-off the denoiser's
# own firefly clamp already makes at the image level (project_denoiser_
# firefly_clamp memory), just applied earlier, at the estimator level,
# where it actually stops the compounding instead of painting over it
# after the fact. 32x is generous relative to any well-behaved path's
# throughput (which should hover near 1 after RR, by construction) while
# still cutting off blowups many orders of magnitude larger.
comptime RR_THROUGHPUT_CLAMP: Float32 = 32.0

# Russian roulette after the first bounce, then save PCG state -- the same
# 8-line epilogue every shade_* path ends with. u_rr must be drawn by the
# caller (not inside here) since some callers already drew it earlier in
# the same bounce for other purposes and must not draw a second one.
@always_inline
def _apply_russian_roulette(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    pcg: PCG32,
    u_rr: Float32,
):
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if u_rr < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
            var new_lum = path_ptr[].throughput.luma()
            if new_lum > RR_THROUGHPUT_CLAMP:
                path_ptr[].throughput *= RR_THROUGHPUT_CLAMP / new_lum
    path_ptr[].pcgState = pcg.state


# ── Shared epilogue for all delta/glossy-delta BSDFs (conductor, coated_conductor,
# dielectric, thin_dielectric): apply the sample, update throughput/bounce
# bookkeeping, run Russian roulette, and save PCG state. Each material's shade_*
# wrapper only differs in how it builds (bs, hit_point) — this is everything that
# happens once those are known. bs.is_valid is always 1 for the two dielectric
# variants, so the early-return there is a harmless no-op for them.
@always_inline
def _finish_delta_bounce(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    mut pcg: PCG32,
    bs: BxDFSample,
    hit_point: Vec3f,
    default_albedo: RGB,
):
    if bs.is_valid == Int8(0):
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = default_albedo
    path_ptr[].throughput *= bs.f
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)
    path_ptr[].bounce += 1

    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, pcg, u_rr)


# ── Dielectric (glass) branch ─────────────────────────────────────────────────
@always_inline
def shade_dielectric[use_gpu: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    mat: Material_C,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin] = UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling(),
    n_textures: Int = 0,
):
    var (ok, is_sphere, geom_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, meshes, spheres)
    if not ok:
        path_ptr[].active = 0
        return
    if not is_sphere:
        # Smooth shading normal, oriented to the geometric/winding normal (PBRT's
        # FaceForward(Ns, Ng)): the winding — flipped at parse time by
        # ReverseOrientation — is the authoritative outside direction the dielectric
        # uses to pick entering vs exiting. (Using the raw vertex normal instead
        # breaks meshes whose vertex normals point inward, causing spurious TIR.)
        geom_normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geom_normal)
    else:
        # `_hit_geom`/`_sphere_geom_normal_and_ray` returns a FACE-FORWARDED
        # normal (always flipped to oppose the incoming ray) -- correct for
        # diffuse/conductor shading, but it destroys the entering/exiting
        # distinction dielectric needs: `bxdf_sample_dielectric`'s `facing`
        # check against a normal that's ALWAYS pre-flipped toward the ray is
        # tautologically always true, so every hit (both the true entry AND
        # the true exit on the far side of a solid sphere) got read as an
        # "entering" event -- eta=1/ior applied TWICE instead of 1/ior then
        # ior, losing (1/ior)^2/ior^2 = 1/ior^4 of the transmitted radiance
        # for a full traversal instead of the correct factor of 1. Recompute
        # the TRUE, un-flipped outward normal (mirrors the mesh branch's own
        # "dielectric needs the raw pre-faceforward normal" rule, see the
        # GeomContext-builders comment a few hundred lines down) before any
        # normal-map perturbation, which itself also needs the true outward
        # direction as its base frame.
        var sph = spheres[Int(inter.primId.id1)]
        var center = Vec3f(sph.center.x, sph.center.y, sph.center.z)
        var hit_point_raw = ray_org + ray_dir * inter.tHit
        var true_normal = hit_point_raw - center
        var tn_len2 = dot(true_normal, true_normal)
        if tn_len2 > Float32(0.0):
            true_normal = true_normal * (Float32(1.0) / sqrt(tn_len2))
        geom_normal = true_normal
        if mat.normal_tex_idx >= Int32(0):
            # Mitsuba "normalmap" wrapping a dielectric (e.g. the SMS paper's
            # own sphere_sms.xml bumpy glass sphere): perturb the shading
            # normal used for refraction itself, not just NEE shading --
            # this is what turns a smooth-lens refraction into the swirly
            # caustic pattern the reference renderer shows. See
            # _apply_normal_map_sphere's own docstring for the analytic
            # sphere UV parameterization this needs (Sphere_C has none
            # built in).
            geom_normal = _apply_normal_map_sphere[use_gpu](mat, center, hit_point_raw, geom_normal, tex_filenames, textures, n_textures)

    var ior = mat.albedo.r
    # A camera/primary ray (bounce 0) from an exterior camera always enters the
    # glass from air. Some meshes in this model have inward-facing normals (no
    # ReverseOrientation) which would otherwise be read as "exiting" and total-
    # internal-reflect the envmap. Trust the physics for the first bounce.
    var force_entering = path_ptr[].bounce == 0

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, normal) = bxdf_sample_dielectric(geom_normal, ray_dir, ior, force_entering, pcg.next_float())

    var is_reflect = (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0
    var offset = (normal if is_reflect else -normal) * Float32(0.0001)
    var hit_point = ray_org + ray_dir * inter.tHit + offset
    _finish_delta_bounce(path_ptr, pcg, bs, hit_point, RGB(Float32(1)))

# Thin dielectric (type 9): one-sided glass — Fresnel selects reflect or transmit,
# but transmitted ray is NOT refracted (direction unchanged). Models window glass,
# soap bubbles, thin films. IOR stored in mat.albedo.r like regular dielectric.
@always_inline
def shade_thin_dielectric(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    mat: Material_C,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
):
    var (ok, is_sphere, geom_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, meshes, spheres)
    if not ok:
        path_ptr[].active = 0
        return

    var ior = mat.albedo.r

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, normal) = bxdf_sample_thin_dielectric(geom_normal, ray_dir, ior, pcg.next_float())

    var is_reflect = (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0
    var offset = (normal if is_reflect else -normal) * Float32(0.0001)
    var hit_point = ray_org + ray_dir * inter.tHit + offset
    _finish_delta_bounce(path_ptr, pcg, bs, hit_point, RGB(Float32(1)))


# ── Conductor (mirror + GGX microfacet) branch ────────────────────────────────

@always_inline
def _nee_loop_simple[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    alpha: Float32,
    mat_kind: Int32,
    wo: Vec3f,
    mut pcg: PCG32,
    guide_write: GuideGrid = null_guide(),
):
    """Distant + point + sphere-light NEE for a 'simple' BxDF (flat
    mat_kind/alb/alpha, evaluable via bxdf_eval_any/_nee_weight_simple_via_spectral)
    -- the one light-loop shape shared VERBATIM between diffuse (mat_kind=0)
    and conductor (mat_kind=1), previously hand-copy-pasted once per material
    (3 loops x 2 materials). Compile-time specialized per call site (no
    runtime indirection), so this is safe to use from GPU-compiled code too.

    Area and infinite lights deliberately stay OUTSIDE this loop: each
    material samples them with a genuinely different strategy (diffuse's
    MNEE-capable _nee_area_lights / cosine-hemisphere _nee_infinite_light
    fallback vs conductor's plain uniform-sphere _sample_area_light_nee/
    _sample_infinite_light_nee) — folding those in here would be a behavior
    change, not a refactor. See project_light_bxdf_interfaces memory."""
    for dl_i in range(ctx.lights.distant_count):
        var ls_d = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i])
        var w_d = _nee_weight_simple_via_spectral(ls_d, mat_kind, alb, alpha, normal, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths)
        if not w_d.is_black():
            var contrib_d = path_ptr[].throughput * w_d
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_d.wi, ls_d.dist, contrib_d, guide_write)

    for pl_i in range(ctx.lights.point_count):
        var ls_p = _sample_point_light_nee(ctx.lights.point_lights[pl_i], hit_point)
        var w_p = _nee_weight_simple_via_spectral(ls_p, mat_kind, alb, alpha, normal, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths)
        if not w_p.is_black():
            var contrib_p = path_ptr[].throughput * w_p
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_p.wi, ls_p.dist * Float32(0.9999), contrib_p, guide_write)

    for sph_i in range(ctx.lights.sphere_count):
        var ls_sph = _sample_sphere_light_nee(ctx.lights.spheres[sph_i], ctx.lights.sphere_count, hit_point, pcg)
        var w_sph = _nee_weight_simple_via_spectral(ls_sph, mat_kind, alb, alpha, normal, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths)
        if not w_sph.is_black():
            var contrib_sph = path_ptr[].throughput * w_sph
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_sph.wi, ls_sph.dist * Float32(0.9999), contrib_sph, guide_write)


@always_inline
def _shade_conductor_nee[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    n: Vec3f,
    wo: Vec3f,
    hit_point: Vec3f,
    f0: RGB,
    alpha: Float32,
    mut pcg: PCG32,
):
    """Distant + point + infinite-light NEE for a ROUGH (non-delta) conductor
    bounce — isotropic GGX approximation (alpha = max(roughU,roughV), same
    simplification bdpt.mojo/sppm.mojo's own conductor connection/gather use
    for anisotropic materials; see bxdf_eval_conductor_ggx's docstring).
    Only called when bxdf_is_delta(bs.flags) is False — a true mirror bounce
    has zero probability of reflecting any finite-solid-angle/delta light
    except along the exact mirror direction, so NEE is skipped entirely for
    those (matches _finish_delta_bounce's existing behavior).

    Distant/point/sphere lights need MIS exactly as bxdf_eval_any/
    _nee_weight_simple already handle it (none for delta distant/point, power
    heuristic against bxdf_pdf_conductor_ggx for sphere/infinite) — via the
    shared Light interface (bvh.mojo's LightSample samplers) + BxDF
    interface, replacing 4 formerly hand-inlined blocks also duplicated in
    _shade_diffuse_nee/shade_hair. Sphere-light NEE is new here (this
    material previously had none). Area-light NEE (via _sample_area_light_nee,
    a generic LightSample wrapper around the mesh/curve area-light picker)
    is also new here — conductor/coated_conductor previously had NO
    area-light NEE at all (a pre-existing, explicitly documented gap; see
    project_light_bxdf_interfaces memory)."""
    var ls_area = _sample_area_light_nee(ctx, hit_point, pcg)
    var w_area = _nee_weight_simple_via_spectral(ls_area, Int32(1), f0, alpha, n, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths)
    if not w_area.is_black():
        var contrib_area = path_ptr[].throughput * w_area
        _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_area.wi, ls_area.dist * Float32(0.9999), contrib_area)

    _nee_loop_simple[enqueue_shadow](path_ptr, ctx, n, hit_point, f0, alpha, Int32(1), wo, pcg)

    for inf_i in range(ctx.lights.infinite_count):
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var w_e = _nee_weight_simple_via_spectral(ls_e, Int32(1), f0, alpha, n, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths)
        if not w_e.is_black():
            var contrib_e = path_ptr[].throughput * w_e
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_e.wi, ls_e.dist, contrib_e)


@always_inline
def shade_conductor[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    # Texture-driven roughness ("texture roughness"/"uroughness" — see
    # material_builder.mojo): resolve to a local mutable copy of `mat` here,
    # once, before anything reads roughU/V below, rather than threading a
    # texture lookup through every downstream GGX-alpha call site. Isotropic
    # only (see Material_C.rough_tex_idx's docstring); falls back to the
    # parsed scalar roughU/V when there's no texture or no UVs.
    var mat_eff = mat
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres)
    if not ok:
        path_ptr[].active = 0
        return

    var normal: Vec3f
    var tangent: Vec3f
    var bitangent: Vec3f
    var alpha_x: Float32
    var alpha_y: Float32

    if is_sphere:
        # Analytic sphere: exact normal IS the shading normal (no
        # interpolation), no UV space to drive a roughness texture or an
        # anisotropy-aligned tangent -- arbitrary Frisvad frame, same
        # isotropic fallback the triangle path uses when it has no UVs.
        normal = geo_normal
        alpha_x = max(mat_eff.roughU, Float32(0.0001))
        alpha_y = max(mat_eff.roughV, Float32(0.0001))
        var frame0 = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
        tangent = Vec3f(frame0.x.x, frame0.x.y, frame0.x.z)
        bitangent = Vec3f(frame0.y.x, frame0.y.y, frame0.y.z)
    else:
        if mat_eff.rough_tex_idx >= Int32(0) and Int(mesh.uvs) > 4:
            var bw0 = Float32(1.0) - inter.u - inter.v
            var uv_u = bw0*mesh.uvs[v0*2]   + inter.u*mesh.uvs[v1*2]   + inter.v*mesh.uvs[v2*2]
            var uv_v = bw0*mesh.uvs[v0*2+1] + inter.u*mesh.uvs[v1*2+1] + inter.v*mesh.uvs[v2*2+1]
            var found = False
            var rtex = sample_texture[use_gpu](Int(mat_eff.rough_tex_idx), uv_u, uv_v, True, Float32(0.0),
                ctx.tex_filenames, ctx.textures, ctx.n_textures, found)
            if found:
                # Perceptual roughness -> GGX alpha (remaproughness=true default,
                # matching the scalar-float "roughness" param's own sqrt() remap
                # in material_builder.mojo -- not threading the remaproughness
                # bool through the texture path since every scene seen so far
                # uses the default).
                var r = sqrt(max(rtex.luma(), Float32(0.0)))
                mat_eff.roughU = r
                mat_eff.roughV = r

        var p0 = Vec3f(mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
        var p1 = Vec3f(mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
        var p2 = Vec3f(mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

        # Use interpolated shading normal for smooth specular reflections
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal)

        # roughU/V already hold the resolved GGX alpha — no squaring here.
        alpha_x = max(mat_eff.roughU, Float32(0.0001))
        alpha_y = max(mat_eff.roughV, Float32(0.0001))

        # Anisotropy tangent frame: UV-gradient (aligned to texture space) when the
        # mesh has UVs and the material is anisotropic; else an arbitrary Frisvad
        # frame (isotropic GGX / perfect mirror don't care about tangent direction).
        if Int(mesh.uvs) > 4 and alpha_x != alpha_y:
            var dp1 = p1 - p0; var dp2 = p2 - p0
            var u0f = mesh.uvs[v0*2]; var v0f = mesh.uvs[v0*2+1]
            var u1f = mesh.uvs[v1*2]; var v1f = mesh.uvs[v1*2+1]
            var u2f = mesh.uvs[v2*2]; var v2f = mesh.uvs[v2*2+1]
            var du1 = u1f - u0f; var dv1 = v1f - v0f
            var du2 = u2f - u0f; var dv2 = v2f - v0f
            var det = du1 * dv2 - du2 * dv1
            if det != Float32(0.0):
                var inv_det = Float32(1.0) / det
                tangent = (dp1 * dv2 - dp2 * dv1) * inv_det
                var tlen = dot(tangent, tangent)
                if tlen > Float32(0.0): tangent = tangent * (Float32(1.0) / sqrt(tlen))
                bitangent = cross(normal, tangent)
                var blen = dot(bitangent, bitangent)
                if blen > Float32(0.0): bitangent = bitangent * (Float32(1.0) / sqrt(blen))
            else:
                var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
                tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
                bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
        else:
            var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
            tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
            bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

    var hit_point = ray_org + ray_dir * inter.tHit + geo_normal * Float32(0.0001)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
    var gc = GeomContext(normal, geo_normal, hit_point, wo, tangent, bitangent,
        RGB(Float32(0.0)), Float32(0.0))

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var bs = bxdf_sample_conductor(gc, mat_eff, pcg.next_float(), pcg.next_float())
    if bs.is_valid != Int8(0) and not bxdf_is_delta(bs.flags):
        var alpha_iso = max(alpha_x, alpha_y)
        _shade_conductor_nee[enqueue_shadow](path_ptr, ctx, normal, wo, hit_point, mat_eff.albedo, alpha_iso, pcg)
        path_ptr[].pcgState = pcg.state
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].albedo = mat_eff.albedo
        path_ptr[].throughput *= bs.f
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = bxdf_pdf_conductor_ggx(normal, wo, bs.wi, alpha_iso)
        path_ptr[].bounce += 1
        var u_rr = pcg.next_float()
        _apply_russian_roulette(path_ptr, pcg, u_rr)
    else:
        _finish_delta_bounce(path_ptr, pcg, bs, hit_point, mat_eff.albedo)


# ── Measured (tabulated Dupuy & Jakob BxDF) ──────────────────────────────────
# The real MeasuredBxDF, not an approximation — see measured_bsdf.mojo's
# loader + measured_bxdf_eval.mojo's f/Sample_f/PDF port. Always glossy
# reflection (never delta, never transmissive), isotropic in practice (see
# the loader's isotropic-only scope note), so an arbitrary Frisvad tangent
# frame is fine — no UV alignment needed (measured materials aren't
# anisotropic in this scene corpus).
#
# Deliberately NOT @always_inline, unlike every sibling _shade_X_nee (e.g.
# _shade_conductor_nee) -- matching _shade_diffuse_nee's precedent (the other
# NEE helper in this file heavy enough to warrant it). This is a genuine
# code-size win kept on its own merits, but it turned out NOT to be the fix
# for shade_measured_gpu's CUDA_ERROR_INVALID_PTX (see below).
#
# Takes `measured_brdfs` + an index rather than `mb: MeasuredBRDF_C` (loading
# mb LOCALLY below instead) to avoid MeasuredBRDF_C -- a TrivialRegisterPassable
# struct with 12 pointer fields -- crossing this now-real call boundary by
# value, the modular/modular#6759 hazard already documented on SpectralHandle
# (see spectrum.mojo). This was tried as a fix candidate before the real bug
# (below) was found and did NOT resolve it on its own -- but it's a
# reasonable precaution independent of that, so it's kept.
#
# The ACTUAL root cause (root-caused 2026-07-10 via progressive stubbing of
# _shade_measured_nee's body down to single expressions): `_bxdf_eval_measured_core`
# used plain `std.math.atan2` for phi_o/phi_m, which depends on an unresolved
# CUDA libdevice extern and cannot produce valid PTX on this machine's
# toolchain -- see `_atan2f` in bvh.mojo ("avoids the unresolved libdevice
# extern on GPU"), already used for hair's own phi computation for exactly
# this reason. Fixed by switching measured_bxdf_eval.mojo's atan2 calls to
# `_atan2f`. See project_gpu_ptx_environment_break memory for the full
# bisection trail, including the earlier wrong kernel-size and by-value-struct
# hypotheses.
def _shade_measured_nee[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutExternalOrigin],
    measured_idx: Int32,
    tangent: Vec3f, bitangent: Vec3f, normal: Vec3f,
    wo: Vec3f,
    hit_point: Vec3f,
    mut pcg: PCG32,
):
    """Distant + point + sphere + area + infinite-light NEE for a measured
    surface — same 5-light-type shape as _shade_conductor_nee, with
    _nee_weight_measured in place of _nee_weight_simple_via_spectral (which
    can't take a MeasuredBRDF_C, same reason hair has its own NEE)."""
    var mb = measured_brdfs[Int(measured_idx)]
    var ls_area = _sample_area_light_nee(ctx, hit_point, pcg)
    var w_area = _nee_weight_measured(ls_area, mb, tangent, bitangent, normal, wo, path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
    if not w_area.is_black():
        var contrib_area = path_ptr[].throughput * w_area
        _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_area.wi, ls_area.dist * Float32(0.9999), contrib_area)

    for dl_i in range(ctx.lights.distant_count):
        var ls_d = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i])
        var w_d = _nee_weight_measured(ls_d, mb, tangent, bitangent, normal, wo, path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
        if not w_d.is_black():
            var contrib_d = path_ptr[].throughput * w_d
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_d.wi, ls_d.dist, contrib_d)

    for pl_i in range(ctx.lights.point_count):
        var ls_p = _sample_point_light_nee(ctx.lights.point_lights[pl_i], hit_point)
        var w_p = _nee_weight_measured(ls_p, mb, tangent, bitangent, normal, wo, path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
        if not w_p.is_black():
            var contrib_p = path_ptr[].throughput * w_p
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_p.wi, ls_p.dist * Float32(0.9999), contrib_p)

    for sph_i in range(ctx.lights.sphere_count):
        var ls_sph = _sample_sphere_light_nee(ctx.lights.spheres[sph_i], ctx.lights.sphere_count, hit_point, pcg)
        var w_sph = _nee_weight_measured(ls_sph, mb, tangent, bitangent, normal, wo, path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
        if not w_sph.is_black():
            var contrib_sph = path_ptr[].throughput * w_sph
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_sph.wi, ls_sph.dist * Float32(0.9999), contrib_sph)

    for inf_i in range(ctx.lights.infinite_count):
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var w_e = _nee_weight_measured(ls_e, mb, tangent, bitangent, normal, wo, path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
        if not w_e.is_black():
            var contrib_e = path_ptr[].throughput * w_e
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_e.wi, ls_e.dist, contrib_e)

@always_inline
def shade_measured[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres)
    if not ok:
        path_ptr[].active = 0
        return

    var normal: Vec3f
    var hit_normal = geo_normal

    if is_sphere:
        # Analytic sphere: exact normal, no interpolation artifact so the
        # silhouette-grazing fallback below doesn't apply.
        normal = geo_normal
    else:
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal)

        # Silhouette-grazing shading-normal fallback: geo_normal is guaranteed
        # face-forwarded toward wo (_geom_normal_and_ray), but the INTERPOLATED
        # shading normal isn't -- on a curved/low-poly mesh viewed near-edge-on
        # (e.g. the car body silhouette), it can end up on the opposite side of
        # wo from geo_normal even though _shading_normal aligned it to
        # geo_normal's hemisphere overall. When that happens, wo lands on the
        # "back" (z<0) side of the (tangent,bitangent,normal) frame while a
        # perfectly valid, physically-illuminating light direction wi lands on
        # the front (z>0) side -- MeasuredBxDF's same-hemisphere gate
        # (wo.z*wi.z<=0) then hard-zeros f for every light sample, discarding
        # all direct illumination at exactly these pixels (confirmed against the
        # pbrt reference: gonzales renders solid black here, pbrt shows lit sky
        # reflection). Falling back to geo_normal for shading removes the
        # artifact at its root, matching the standard fix (pbrt determines
        # reflect/transmit-side membership from the geometric, not shading,
        # normal) without touching the general shading-normal interpolation
        # path other materials still rely on for smooth appearance.
        if dot(normal, -ray_dir) <= Float32(0.0):
            normal = geo_normal

    var hit_point = ray_org + ray_dir * inter.tHit + hit_normal * Float32(0.0001)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])

    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

    if mat.measured_idx < Int32(0):
        # Load failure fallback (see material_builder.mojo) -- render as flat
        # black rather than dereferencing a nonexistent measured_brdfs entry.
        path_ptr[].active = 0
        return
    var mb = ctx.measured_brdfs[Int(mat.measured_idx)]

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # NEE runs unconditionally, BEFORE the BSDF-sample validity checks below
    # -- matches pbrt's own design, where direct lighting is independent of
    # whether BSDF sampling succeeds (MeasuredBxDF::f() only bails out on a
    # degenerate half-vector, not on "the self-sampled reflection direction
    # happened to land below the horizon"). The tabulated lobe frequently
    # samples a sub-horizon wi near grazing angles (e.g. the front of a car
    # body), and gating NEE on that sample's validity was discarding real,
    # correctly-lit direct illumination at exactly those pixels -- confirmed
    # against the pbrt reference showing they should stay lit, not black.
    _shade_measured_nee[enqueue_shadow](path_ptr, ctx, ctx.measured_brdfs, mat.measured_idx, tangent, bitangent, normal, wo, hit_point, pcg)

    var wo_l = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, normal))
    var (wi_l, f, pdf, valid) = bxdf_sample_measured(mb, wo_l, pcg.next_float(), pcg.next_float(), path_ptr[].wavelengths, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65)
    path_ptr[].pcgState = pcg.state

    if not valid or pdf <= Float32(0.0):
        path_ptr[].active = 0
        return

    var wi = tangent * wi_l[0] + bitangent * wi_l[1] + normal * wi_l[2]
    var wilen = dot(wi, wi)
    if wilen > Float32(0.0):
        wi = wi * (Float32(1.0) / sqrt(wilen))
    var cos_wi = dot(wi, normal)
    if cos_wi <= Float32(0.0):
        path_ptr[].active = 0
        return

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(wi[0], wi[1], wi[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = f
    # General non-delta BxDFSample convention (f already includes the
    # measured BRDF's own 1/AbsCosTheta(wi) term, matching pbrt's own fr —
    # this cos_wi is the separate Monte-Carlo importance-sampling weight,
    # not a duplicate of that internal term; no analytic cancellation exists
    # for a tabulated BxDF the way there is for VNDF-sampled GGX conductor).
    path_ptr[].throughput *= f * (cos_wi / pdf)
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].lastBsdfPdf = bxdf_pdf_measured(mb, wo_l, wi_l)
    path_ptr[].bounce += 1
    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, pcg, u_rr)


# CoatedConductor: dielectric clearcoat over GGX conductor.
# Schlick Fresnel at the air/coat interface: F_schlick(cos_theta, 0, 1, ior)
# selects between coat specular reflection (F) and conducting GGX layer (1-F).
# This is an energy-conserving two-lobe approximation of pbrt's LayeredBxDF.
@always_inline
def shade_coated_conductor[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    # No shading-normal interpolation here (unlike conductor) — matches the
    # pre-existing coated_conductor behavior of using the flat geometric
    # normal (which, for a sphere, IS already the exact shading normal).
    var (ok, is_sphere, normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres)
    if not ok:
        path_ptr[].active = 0
        return
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var ior = mat.emission.r if mat.emission.r > Float32(1.0) else Float32(1.5)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
    # Frisvad frame — isotropic GGX only (single alpha), no anisotropy alignment needed.
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
    var gc = GeomContext(normal, normal, hit_point, wo, tangent, bitangent,
        RGB(Float32(0.0)), Float32(0.0))

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var bs = bxdf_sample_coated_conductor(gc, mat, ior, pcg.next_float(), pcg.next_float(), pcg.next_float())
    if bs.is_valid != Int8(0) and not bxdf_is_delta(bs.flags):
        # Approximation (matches bdpt.mojo/sppm.mojo's own coated_conductor
        # connection/gather treatment): NEE reuses conductor's own GGX eval/
        # F0, ignoring the coat's own (1-f_coat) attenuation and its separate
        # luma-Fresnel blend. alpha = (roughU+roughV)/2, matching
        # bxdf_sample_coated_conductor's own internal averaging.
        var alpha_cc = (max(mat.roughU, Float32(0.0001)) + max(mat.roughV, Float32(0.0001))) * Float32(0.5)
        _shade_conductor_nee[enqueue_shadow](path_ptr, ctx, normal, wo, hit_point, mat.albedo, alpha_cc, pcg)
        path_ptr[].pcgState = pcg.state
        path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].albedo = mat.albedo
        path_ptr[].throughput *= bs.f
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = bxdf_pdf_conductor_ggx(normal, wo, bs.wi, alpha_cc)
        path_ptr[].bounce += 1
        var u_rr = pcg.next_float()
        _apply_russian_roulette(path_ptr, pcg, u_rr)
    else:
        _finish_delta_bounce(path_ptr, pcg, bs, hit_point, mat.albedo)


# Mix material: randomly select one of two sub-materials using amount as probability.
# Sub-material indices are packed into mat.tex_idx: low 16 bits = idx1, high 16 bits = idx2.
# mat.roughU = blend amount (probability of picking mat2).
# Not @always_inline: shade_mix ↔ _shade_dispatch would form an always_inline recursion.
def shade_mix[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var packed = mat.tex_idx
    var idx1 = Int(packed & Int32(0xFFFF))
    var idx2 = Int((packed >> 16) & Int32(0xFFFF))
    var amount = mat.roughU  # blend factor: 0 = all mat1, 1 = all mat2
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var chosen_idx = idx2 if pcg.next_float() < amount else idx1
    path_ptr[].pcgState = pcg.state
    var sub_mat = ctx.materials[chosen_idx]
    if sub_mat.type == Int8(8):
        sub_mat.type = Int8(1)  # guard against mix-of-mix cycle
    _shade_dispatch[use_gpu, enqueue_shadow](sub_mat, path_ptr, inter, ctx)

# Apply tangent-space normal map: samples the texture, decodes [-1,1] normal,
# rotates it into world space via UV-gradient tangent frame.
# Returns geom_normal unchanged when normal_tex_idx < 0 or no UVs.
@always_inline
def _apply_normal_map[use_gpu: Bool](
    mat: Material_C,
    v0: Int, v1: Int, v2: Int,
    mesh: TriangleMesh_C,
    inter: Intersection_C,
    geom_normal: Vec3f,
    p0: Vec3f,
    p1: Vec3f,
    p2: Vec3f,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures: Int,
    pixel_uv: Float32 = Float32(0.0),
) -> Vec3f:
    if mat.normal_tex_idx < Int32(0) or Int(mesh.uvs) <= 4:
        return geom_normal
    # Compute barycentric UV coordinates
    var dp1 = p1 - p0; var dp2 = p2 - p0
    var u0f = mesh.uvs[v0*2]; var v0f = mesh.uvs[v0*2+1]
    var u1f = mesh.uvs[v1*2]; var v1f = mesh.uvs[v1*2+1]
    var u2f = mesh.uvs[v2*2]; var v2f = mesh.uvs[v2*2+1]
    var du1 = u1f - u0f; var dv1 = v1f - v0f
    var du2 = u2f - u0f; var dv2 = v2f - v0f
    var det = du1 * dv2 - du2 * dv1
    if det == Float32(0.0):
        return geom_normal
    var inv_det = Float32(1.0) / det
    # Tangent = dP/du; bitangent = cross(N, tangent). This matches pbrt's shading
    # frame Frame::FromXZ(dpdu, n) (whose Y axis is cross(n, dpdu)) — NOT dP/dv,
    # which has the opposite handedness here and mirrors the relief in a grazing
    # view (the seam this once showed was the view-ray faceforward bug, fixed
    # separately).
    var tangent = (dp1 * dv2 - dp2 * dv1) * inv_det
    var tlen = dot(tangent, tangent)
    if tlen <= Float32(0.0): return geom_normal
    tangent = tangent * (Float32(1.0) / sqrt(tlen))
    # pbrt's frame is FromXZ(dpdu, n): the bitangent is cross(n, dpdu) (matches
    # materials.h NormalMap), NOT dP/dv.
    var bitangent = cross(geom_normal, tangent)
    var blen = dot(bitangent, bitangent)
    if blen <= Float32(0.0): return geom_normal
    bitangent = bitangent * (Float32(1.0) / sqrt(blen))
    # Interpolate UV at the hit point using the barycentrics (inter.u, inter.v).
    # (A centroid approximation here makes the normal map constant per triangle,
    #  so bumps vanish on low-poly meshes like a 2-triangle floor.)
    var bw0 = Float32(1.0) - inter.u - inter.v
    var uv_u = bw0 * u0f + inter.u * u1f + inter.v * u2f
    var uv_v = bw0 * v0f + inter.u * v1f + inter.v * v2f  # unflipped; sample_texture applies the V-flip
    # Sample the normal map raw (no sRGB) and decode [0,1] -> [-1,1].
    var found = False
    var ns = sample_texture[use_gpu](Int(mat.normal_tex_idx), uv_u, uv_v, True, pixel_uv, tex_filenames, textures, n_textures, found)
    if found:
        var nx = ns.r * Float32(2.0) - Float32(1.0)
        var ny = ns.g * Float32(2.0) - Float32(1.0)
        var nz = ns.b * Float32(2.0) - Float32(1.0)
        # Rotate tangent-space normal to world space
        var world_n = tangent * nx + bitangent * ny + geom_normal * nz
        var wn_len = dot(world_n, world_n)
        if wn_len > Float32(0.0):
            return world_n * (Float32(1.0) / sqrt(wn_len))
    return geom_normal

@always_inline
def _apply_normal_map_sphere[use_gpu: Bool](
    mat: Material_C,
    center: Vec3f,
    hit_point: Vec3f,
    geom_normal: Vec3f,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures: Int,
) -> Vec3f:
    """Analytic-sphere counterpart to `_apply_normal_map` -- no mesh/UVs
    exist for a `Sphere_C`, so u/v and the tangent frame are derived from
    the standard spherical parameterization instead of barycentrics.
    Matches Mitsuba's OWN `Sphere::compute_surface_interaction` convention
    exactly (`src/shapes/sphere.cpp`: `theta = unit_angle_z(local)` --
    polar angle from +Z, not +Y -- `phi = atan2(local.y, local.x)`,
    `dp_du = (-local.y, local.x, 0) * 2*pi`, `dp_dv = (local.z*cos_phi,
    local.z*sin_phi, -rd) * pi` where `rd = hypot(local.x, local.y)`) --
    NOT the pbrt/y-up convention this originally used, which put the
    swirl pattern from a normal map at the wrong place on the sphere
    (right UV topology, wrong pole axis, so the whole pattern sat rotated
    90 degrees relative to a real Mitsuba render of the same scene). Only
    correct when the sphere's own object-to-world transform is identity
    (true for `sphere_sms.xml` and any other `<shape type="sphere">` with
    no `<transform>` block) -- `Sphere_C` stores no orientation, so a
    rotated sphere's local Z axis can't be recovered here; out of scope,
    same "no non-uniform scale" limitation `_mit_process_sphere` already
    documents.
    Degenerates to a zero tangent at the poles (`rd == 0`) like any sphere
    UV parameterization -- falls back to an arbitrary tangent frame there
    via `Frame.from_z`, matching how `_build_geom_context_full`'s own
    sphere branch already builds a frame with no meaningful tangent
    direction to prefer."""
    if mat.normal_tex_idx < Int32(0):
        return geom_normal
    var local = hit_point - center
    var rd2 = local[0]*local[0] + local[1]*local[1]
    var rd = sqrt(rd2)
    var theta = _atan2f(rd, local[2])
    var phi = _atan2f(local[1], local[0])
    if phi < Float32(0.0):
        phi += Float32(2.0) * Float32(3.14159265358979)
    var u = phi / (Float32(2.0) * Float32(3.14159265358979))
    var v = theta / Float32(3.14159265358979)
    var dp_du = Vec3f(-local[1], local[0], Float32(0.0)) * (Float32(2.0) * Float32(3.14159265358979))
    var dp_dv: Vec3f
    if rd > Float32(1e-8):
        var inv_rd = Float32(1.0) / rd
        var cos_phi = local[0] * inv_rd
        var sin_phi = local[1] * inv_rd
        dp_dv = Vec3f(local[2] * cos_phi, local[2] * sin_phi, -rd) * Float32(3.14159265358979)
    else:
        dp_dv = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0)) * Float32(3.14159265358979)
    var tangent: Vec3f
    var du_len2 = dot(dp_du, dp_du)
    if du_len2 > Float32(1e-12):
        tangent = dp_du * (Float32(1.0) / sqrt(du_len2))
    else:
        var frame = Frame.from_z(Vec3f(geom_normal[0], geom_normal[1], geom_normal[2]))
        tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = cross(geom_normal, tangent)
    var b_len2 = dot(bitangent, bitangent)
    if b_len2 <= Float32(1e-12):
        return geom_normal
    bitangent = bitangent * (Float32(1.0) / sqrt(b_len2))
    var found = False
    # `sample_texture` applies pbrt's V-flip (tv = 1 - v) because a pbrt mesh
    # puts V=0 at the TOP of the image. This sphere's (u, v) are built in
    # MITSUBA's convention instead (see this function's own docstring), so
    # that flip is wrong here and has to be pre-cancelled -- otherwise the
    # normal map is read vertically MIRRORED, which tilts the shading normal
    # by ~11 degrees everywhere and bends every refracted ray.
    #
    # Verified against the reference renderer directly, at a matched hit
    # point (its own `normalmap.cpp::frame()` instrumented to dump
    # p/uv/n_local): our (u, v) already reproduce its uv to 5 decimals, and
    # sampling the map at v matches its shading normal to 0.00 degrees while
    # sampling at 1 - v is off by 11.53 degrees. Only the analytic-sphere
    # path is affected -- a pbrt trianglemesh normal map goes through
    # `_apply_normal_map`, which genuinely wants pbrt's flip.
    var ns = sample_texture[use_gpu](Int(mat.normal_tex_idx), u, Float32(1.0) - v, True, Float32(0.0), tex_filenames, textures, n_textures, found)
    if not found:
        return geom_normal
    var nx = ns.r * Float32(2.0) - Float32(1.0)
    var ny = ns.g * Float32(2.0) - Float32(1.0)
    var nz = ns.b * Float32(2.0) - Float32(1.0)
    var world_n = tangent * nx + bitangent * ny + geom_normal * nz
    var wn_len = dot(world_n, world_n)
    if wn_len > Float32(0.0):
        return world_n * (Float32(1.0) / sqrt(wn_len))
    return geom_normal


# ── Geometry context builders ─────────────────────────────────────────────────
# There's no _build_geom_context_minimal: each delta BSDF's geometry needs turned
# out to be materially different (conductor needs a UV-gradient anisotropy tangent
# unavailable here; coated_conductor skips shading-normal interpolation; dielectric/
# thin_dielectric need the raw pre-faceforward normal for entering/exiting), so
# shade_conductor/shade_coated_conductor build their GeomContext inline instead of
# sharing one minimal builder.

# Full context for NEE materials (diffuse, diffuse_transmit, coated_diffuse).
# Computes pixel_uv, applies normal map, looks up albedo texture.
# hit_point offset uses the bumped shading normal (matches shade_diffuse convention).
# Does NOT perform the backface check — caller must check dot(gc.normal, gc.wo) > 0.
@always_inline
def _build_geom_context_full[use_gpu: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    mat: Material_C,
    ctx: ShadeContext,
) -> Tuple[GeomContext, Bool]:
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances)
    if not ok:
        var z = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (GeomContext(z, z, z, z, z, z, RGB(Float32(0.0)), Float32(0.0)), False)

    if is_sphere:
        # Analytic sphere: exact normal, no UVs/texture/normal-map to resolve
        # -- alb falls back to the material's flat albedo (see project_mitsuba_parser
        # memory: no sphere UV parameterization exists in this codebase yet).
        var hit_point = ray_org + ray_dir * inter.tHit + geo_normal * Float32(0.0001)
        var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
        var frame = Frame.from_z(Vec3f(geo_normal[0], geo_normal[1], geo_normal[2]))
        var tangent = Vec3f(frame.x.x, frame.x.y, frame.x.z)
        var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
        return (GeomContext(geo_normal, geo_normal, hit_point, wo, tangent, bitangent, mat.albedo, Float32(0.0)), True)

    var ng_ff = geo_normal
    var p0 = Vec3f(mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = Vec3f(mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = Vec3f(mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var pixel_uv = Float32(0.0)
    if ctx.px_scale > Float32(0.0) and Int(mesh.uvs) > 1:
        var fu1 = mesh.uvs[v1*2] - mesh.uvs[v0*2]; var fv1 = mesh.uvs[v1*2+1] - mesh.uvs[v0*2+1]
        var fu2 = mesh.uvs[v2*2] - mesh.uvs[v0*2]; var fv2 = mesh.uvs[v2*2+1] - mesh.uvs[v0*2+1]
        var det = fu1*fv2 - fu2*fv1
        if det != Float32(0.0):
            var inv = Float32(1.0) / det
            var dpdu = (p1 - p0) * (fv2 * inv) - (p2 - p0) * (fv1 * inv)
            var dpdu_len = sqrt(dot(dpdu, dpdu))
            if dpdu_len > Float32(0.0):
                var rc = dot(ng_ff, ray_dir)
                if rc < Float32(0.0): rc = -rc
                if rc < Float32(0.05): rc = Float32(0.05)
                pixel_uv = (inter.tHit * ctx.px_scale / rc) / dpdu_len

    var normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal, inter.primId.instanceIdx, ctx.instances)
    normal = _apply_normal_map[use_gpu](mat, v0, v1, v2, mesh, inter, normal, p0, p1, p2,
        ctx.tex_filenames, ctx.textures, ctx.n_textures, pixel_uv)
    if dot(normal, ng_ff) < Float32(0.0):
        normal = -normal

    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures, pixel_uv)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)
    return (GeomContext(normal, ng_ff, hit_point, wo, tangent, bitangent, alb, pixel_uv), True)

# Extract 8 consecutive Sobol dimensions for one non-delta bounce and advance
# the path's sampler_dim counter.
@always_inline
def _draw_sobol_8(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
) -> SobolSamples8:
    var _sidx = Int(path_ptr[].sobol_idx)
    var _sdim = Int(path_ptr[].sampler_dim)
    var _sinc = path_ptr[].pcgInc
    var u_light = sobol_sample(_sidx, _sdim + 0, mix_bits_u64(_sinc ^ UInt64(_sdim + 0)), sobol_matrices)
    var u_bary1 = sobol_sample(_sidx, _sdim + 1, mix_bits_u64(_sinc ^ UInt64(_sdim + 1)), sobol_matrices)
    var u_bary2 = sobol_sample(_sidx, _sdim + 2, mix_bits_u64(_sinc ^ UInt64(_sdim + 2)), sobol_matrices)
    var u_env1  = sobol_sample(_sidx, _sdim + 3, mix_bits_u64(_sinc ^ UInt64(_sdim + 3)), sobol_matrices)
    var u_env2  = sobol_sample(_sidx, _sdim + 4, mix_bits_u64(_sinc ^ UInt64(_sdim + 4)), sobol_matrices)
    var u_scat1 = sobol_sample(_sidx, _sdim + 5, mix_bits_u64(_sinc ^ UInt64(_sdim + 5)), sobol_matrices)
    var u_scat2 = sobol_sample(_sidx, _sdim + 6, mix_bits_u64(_sinc ^ UInt64(_sdim + 6)), sobol_matrices)
    var u_rr    = sobol_sample(_sidx, _sdim + 7, mix_bits_u64(_sinc ^ UInt64(_sdim + 7)), sobol_matrices)
    path_ptr[].sampler_dim += Int32(8)
    return SobolSamples8(u_light, u_bary1, u_bary2, u_env1, u_env2, u_scat1, u_scat2, u_rr)

# ── Hair BSDF helpers ────────────────────────────────────────────────────────
# Live in bvh.mojo, not here — sppm.mojo/bdpt.mojo need them too (for
# connectible-vertex/gather/NEE evaluation of a stored hair vertex), and
# shading.mojo already imports FROM sppm.mojo, so sppm.mojo can't import back
# from shading.mojo (same import-cycle constraint _mk_sd_full's own move
# documented). See bvh.mojo's _hair_precompute/_hair_eval_lobes/
# _hair_sample_dir/HairLobeConstants.

# ── Marschner/Chiang hair BSDF (3-lobe: R, TT, TRT) ─────────────────────────
def shade_hair[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    """3-lobe hair BSDF (Marschner R/TT/TRT). Material packing:
       albedo = sigma_a (RGB absorption), emission.r = eta (IOR),
       roughU = betaM (longitudinal), roughV = betaN (azimuthal)."""



    # ── Steps 1-10: geometry + optical + per-lobe precompute ─────────────────
    # h and v_global come straight from intersect_curve (geometry.mojo) via
    # Intersection_C.u/.v — no tessellated mesh, no barycentric interpolation.
    # Delegated to bvh.mojo's _hair_precompute (shared with bdpt.mojo/
    # sppm.mojo's connectible-vertex/gather/NEE evaluation of a stored hair
    # vertex) — same formulas as before this was extracted, just relocated.
    if inter.primId.type != 5:
        path_ptr[].active = 0
        return
    var curve_idx = Int(inter.primId.id1)
    var h = max(Float32(-0.99), min(Float32(0.99), inter.u))
    var v_global = inter.v
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var wo = -ray_dir
    var hc = _hair_precompute(mat, ctx.curves, curve_idx, v_global, h, wo)
    # NEE (all light types) and indirect sampling now go through hc directly
    # (bxdf.mojo's _nee_weight_hair, bvh.mojo's _hair_sample_dir) — only
    # geo_normal is still needed locally, for shadow-ray/hit-point offsets.
    var geo_normal = hc.geo_normal

    # ── Hit point for shadow rays (always offset toward incoming side) ────────
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_base = ray_org + ray_dir * inter.tHit
    var curve_eps = curve_offset_eps(hc.radius)
    var hit_point = hit_base + geo_normal * curve_eps

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # ── NEE for infinite/distant/point/sphere lights, via the shared Light
    # interface (bvh.mojo's LightSample samplers) + BxDF interface
    # (bxdf.mojo's _nee_weight_hair, using the `hc` HairLobeConstants
    # already computed above) — replacing 3 formerly hand-inlined blocks
    # also duplicated in _shade_diffuse_nee/_shade_conductor_nee. Each
    # light's shadow-ray origin still needs hair's own sign-flip offset
    # (toward whichever side of geo_normal wi points), so that glue stays
    # here rather than in the shared weight function. Sphere-light NEE is
    # new (this material previously had none).
    for inf_i in range(ctx.lights.infinite_count):
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var w_e = _nee_weight_hair(ls_e, hc)
        if not w_e.is_black():
            var esign = Float32(1.0) if dot(ls_e.wi, geo_normal) >= Float32(0.0) else Float32(-1.0)
            var eorg = hit_base + geo_normal * curve_eps * esign
            var contrib_e = path_ptr[].throughput * w_e
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, eorg, ls_e.wi, ls_e.dist, contrib_e)

    for dl_i in range(ctx.lights.distant_count):
        var ls_d = _sample_distant_light_nee(ctx.lights.distant_lights[dl_i])
        var w_d = _nee_weight_hair(ls_d, hc)
        if not w_d.is_black():
            var dsign = Float32(1.0) if dot(ls_d.wi, geo_normal) >= Float32(0.0) else Float32(-1.0)
            var dorg = hit_base + geo_normal * curve_eps * dsign
            var contrib_d = path_ptr[].throughput * w_d
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, dorg, ls_d.wi, ls_d.dist, contrib_d)

    for pl_i in range(ctx.lights.point_count):
        var ls_p = _sample_point_light_nee(ctx.lights.point_lights[pl_i], hit_base)
        var w_p = _nee_weight_hair(ls_p, hc)
        if not w_p.is_black():
            var psign = Float32(1.0) if dot(ls_p.wi, geo_normal) >= Float32(0.0) else Float32(-1.0)
            var porg = hit_base + geo_normal * curve_eps * psign
            var contrib_p = path_ptr[].throughput * w_p
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, porg, ls_p.wi, ls_p.dist * Float32(0.9999), contrib_p)

    for sph_i in range(ctx.lights.sphere_count):
        var ls_sph = _sample_sphere_light_nee(ctx.lights.spheres[sph_i], ctx.lights.sphere_count, hit_base, pcg)
        var w_sph = _nee_weight_hair(ls_sph, hc)
        if not w_sph.is_black():
            var sphsign = Float32(1.0) if dot(ls_sph.wi, geo_normal) >= Float32(0.0) else Float32(-1.0)
            var sphorg = hit_base + geo_normal * curve_eps * sphsign
            var contrib_sph = path_ptr[].throughput * w_sph
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, sphorg, ls_sph.wi, ls_sph.dist * Float32(0.9999), contrib_sph)

    # ── Step 15: Indirect sampling ────────────────────────────────────────────
    # Delegated to bvh.mojo's _hair_sample_dir (same lobe-pick + vMF/logistic
    # sampling as before this was extracted, identical RNG draw order) —
    # shared with bdpt.mojo/sppm.mojo's own hair-vertex bounce continuation.
    var (wi_s, f_s, pdf, cos_ti_s2) = _hair_sample_dir(hc, pcg)
    var frs = f_s.r; var fgs = f_s.g; var fbs = f_s.b

    # Solid-angle PDF (remove the embedded /cos_ti_s) — used for MIS when indirect path hits env.
    var pdf_solid_angle = pdf * cos_ti_s2

    # Update path state
    path_ptr[].pcgState = pcg.state
    # Offset scattered ray origin based on which side wi_s exits (avoids self-intersection
    # for transmission rays that cross through the ribbon).
    var scatter_sign = Float32(1.0) if dot(wi_s, geo_normal) >= Float32(0.0) else Float32(-1.0)
    var scatter_org = hit_base + geo_normal * curve_eps * scatter_sign
    path_ptr[].ray = Ray_C(
        Point3f(scatter_org[0], scatter_org[1], scatter_org[2]),
        Vec3f(wi_s[0], wi_s[1], wi_s[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        # Approximate albedo AOV from TT lobe (closest to diffuse color)
        path_ptr[].albedo = hc.A1
    path_ptr[].throughput *= RGB(frs, fgs, fbs) * (Float32(1.0) / pdf)
    path_ptr[].lastBsdfPdf = pdf_solid_angle
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].bounce += 1

    var u_rr_hair = pcg.next_float()
    _apply_russian_roulette(path_ptr, pcg, u_rr_hair)


# 2-vertex MNEE: Newton walk for x0→x1(glass1)→x2(glass2)→x3.
# etas must be pre-computed: eta = ior if entering glass, 1/ior if exiting.
# Returns (converged, x1_f, x2_f, bsdf_product, dx1_dxlight).
@always_inline
def _mnee_walk2(
    x0: Vec3f, x3: Vec3f,
    x1_init: Vec3f, n1: Vec3f,
    dp_du1: Vec3f, dp_dv1: Vec3f, eta1: Float32,
    x2_init: Vec3f, n2: Vec3f,
    dp_du2: Vec3f, dp_dv2: Vec3f, eta2: Float32,
    ldp_du: Vec3f, ldp_dv: Vec3f,
) -> Tuple[Bool, Vec3f, Vec3f, Float32, Float32]:
    var x1 = x1_init
    var x2 = x2_init
    var converged = False
    for _iter in range(20):
        # ── Vertex 0 (x1): wi from x0, wo to x2 ────────────────────────────
        var wi1v = x0 - x1; var wi1l = sqrt(dot(wi1v,wi1v))
        var wo1v = x2 - x1; var wo1l = sqrt(dot(wo1v,wo1v))
        if wi1l < Float32(1e-6) or wo1l < Float32(1e-6): break
        var wi1 = wi1v*(Float32(1)/wi1l); var wo1 = wo1v*(Float32(1)/wo1l)
        if dot(n1,wi1)*dot(n1,wo1) >= Float32(0): break
        var H1v = -(wi1 + wo1*eta1); var H1l = sqrt(dot(H1v,H1v))
        if H1l < Float32(1e-10): break
        var H1 = H1v*(Float32(1)/H1l)
        var s1 = dp_du1 - n1*dot(dp_du1,n1); var s1l = sqrt(dot(s1,s1))
        if s1l < Float32(1e-10): break
        s1 = s1*(Float32(1)/s1l); var t1 = cross(n1,s1)
        var ili1 = Float32(1)/(H1l*wi1l); var ilo1 = eta1/(H1l*wo1l)
        var (a0,b0,c0,cv0) = sms_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
            Vec3f(0),Vec3f(0),dp_du2,dp_dv2,False,True)
        # ── Vertex 1 (x2): wi from x1, wo to x3 ────────────────────────────
        var wi2v = x1 - x2; var wi2l = sqrt(dot(wi2v,wi2v))
        var wo2v = x3 - x2; var wo2l = sqrt(dot(wo2v,wo2v))
        if wi2l < Float32(1e-6) or wo2l < Float32(1e-6): break
        var wi2 = wi2v*(Float32(1)/wi2l); var wo2 = wo2v*(Float32(1)/wo2l)
        if dot(n2,wi2)*dot(n2,wo2) >= Float32(0): break
        var H2v = -(wi2 + wo2*eta2); var H2l = sqrt(dot(H2v,H2v))
        if H2l < Float32(1e-10): break
        var H2 = H2v*(Float32(1)/H2l)
        var s2 = dp_du2 - n2*dot(dp_du2,n2); var s2l = sqrt(dot(s2,s2))
        if s2l < Float32(1e-10): break
        s2 = s2*(Float32(1)/s2l); var t2 = cross(n2,s2)
        var ili2 = Float32(1)/(H2l*wi2l); var ilo2 = eta2/(H2l*wo2l)
        var (a1,b1,c1,cv1) = sms_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
            dp_du1,dp_dv1,Vec3f(0),Vec3f(0),True,False)
        _ = a0; _ = c1
        # ── Convergence ──────────────────────────────────────────────────────
        var err = max(max(abs(cv0[0]),abs(cv0[1])), max(abs(cv1[0]),abs(cv1[1])))
        if err < SMS_SOLVER_THRESHOLD:
            converged = True; break
        # ── Block tridiagonal solve: [b0 c0; a1 b1]*[dx0;dx1]=[cv0;cv1] ────
        var (Li0, det0) = mat22_inv(b0)
        if det0 == Float32(0): break
        var A1 = mat22_mul(a1, Li0)
        var Lk1 = b1 - mat22_mul(A1, c0)
        var (Li1, det1) = mat22_inv(Lk1)
        if det1 == Float32(0): break
        var C1_red = cv1 - mat22_mul_v(A1, cv0)
        var dx1 = mat22_mul_v(Li1, C1_red)
        var dx0 = mat22_mul_v(Li0, cv0 - mat22_mul_v(c0, dx1))
        # ── Step clamp ──────────────────────────────────────────────────────
        var step0 = sqrt(dx0[0]*dx0[0]+dx0[1]*dx0[1]); var ms0 = wo1l*Float32(0.5)
        if step0 > ms0: dx0 = dx0*(ms0/step0)
        var step1 = sqrt(dx1[0]*dx1[0]+dx1[1]*dx1[1]); var ms1 = wo2l*Float32(0.5)
        if step1 > ms1: dx1 = dx1*(ms1/step1)
        x1 = x1 - (dp_du1*dx0[0] + dp_dv1*dx0[1])
        x2 = x2 - (dp_du2*dx1[0] + dp_dv2*dx1[1])
    if not converged:
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    # ── Recompute final vertex quantities for transfer matrix ────────────────
    var wi1v = x0 - x1; var wi1l = sqrt(dot(wi1v,wi1v))
    var wo1v = x2 - x1; var wo1l = sqrt(dot(wo1v,wo1v))
    var wi2v = x1 - x2; var wi2l = sqrt(dot(wi2v,wi2v))
    var wo2v = x3 - x2; var wo2l = sqrt(dot(wo2v,wo2v))
    if wi1l < Float32(1e-8) or wo1l < Float32(1e-8) or wi2l < Float32(1e-8) or wo2l < Float32(1e-8):
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    var wi1 = wi1v*(Float32(1)/wi1l); var wo1 = wo1v*(Float32(1)/wo1l)
    var wi2 = wi2v*(Float32(1)/wi2l); var wo2 = wo2v*(Float32(1)/wo2l)
    var H1v = -(wi1+wo1*eta1); var H1l = sqrt(dot(H1v,H1v))
    var H2v = -(wi2+wo2*eta2); var H2l = sqrt(dot(H2v,H2v))
    if H1l < Float32(1e-10) or H2l < Float32(1e-10):
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    var H1 = H1v*(Float32(1)/H1l); var H2 = H2v*(Float32(1)/H2l)
    var s1 = dp_du1-n1*dot(dp_du1,n1); s1 = s1*(Float32(1)/sqrt(dot(s1,s1))); var t1 = cross(n1,s1)
    var s2 = dp_du2-n2*dot(dp_du2,n2); s2 = s2*(Float32(1)/sqrt(dot(s2,s2))); var t2 = cross(n2,s2)
    var ili1 = Float32(1)/(H1l*wi1l); var ilo1 = eta1/(H1l*wo1l)
    var ili2 = Float32(1)/(H2l*wi2l); var ilo2 = eta2/(H2l*wo2l)
    var (a0f,b0f,c0f,cv0f) = sms_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
        Vec3f(0),Vec3f(0),dp_du2,dp_dv2,False,True)
    var (a1f,b1f,c1f,cv1f) = sms_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
        dp_du1,dp_dv1,Vec3f(0),Vec3f(0),True,False)
    _ = a0f; _ = c1f; _ = cv0f; _ = cv1f
    # ── Block tridiagonal LU factorization for transfer matrix ───────────────
    var (Li0f, det0f) = mat22_inv(b0f)
    if det0f == Float32(0): return (False, x1_init, x2_init, Float32(0), Float32(0))
    var U0 = mat22_mul(Li0f, c0f)
    var Lk1f = b1f - mat22_mul(a1f, U0)
    var (Li1f, det1f) = mat22_inv(Lk1f)
    if det1f == Float32(0): return (False, x1_init, x2_init, Float32(0), Float32(0))
    # ── dc_dlight at x2 (wrt x3 position) ───────────────────────────────────
    var dc_du = (ldp_du - wo2*dot(wo2,ldp_du)) * ilo2
    var dc_dv = (ldp_dv - wo2*dot(wo2,ldp_dv)) * ilo2
    dc_du -= H2*dot(dc_du,H2); dc_du = -dc_du
    dc_dv -= H2*dot(dc_dv,H2); dc_dv = -dc_dv
    var dc_dlight = SIMD[DType.float32, 4](dot(dc_du,s2), dot(dc_dv,s2), dot(dc_du,t2), dot(dc_dv,t2))
    # ── Transfer matrix chain ────────────────────────────────────────────────
    var Tp = SIMD[DType.float32, 4](Float32(0)-Li1f[0]*dc_dlight[0]-Li1f[1]*dc_dlight[2],
        Float32(0)-Li1f[0]*dc_dlight[1]-Li1f[1]*dc_dlight[3],
        Float32(0)-Li1f[2]*dc_dlight[0]-Li1f[3]*dc_dlight[2],
        Float32(0)-Li1f[2]*dc_dlight[1]-Li1f[3]*dc_dlight[3])
    Tp = SIMD[DType.float32, 4](Float32(0)-U0[0]*Tp[0]-U0[1]*Tp[2], Float32(0)-U0[0]*Tp[1]-U0[1]*Tp[3],
        Float32(0)-U0[2]*Tp[0]-U0[3]*Tp[2], Float32(0)-U0[2]*Tp[1]-U0[3]*Tp[3])
    var dx1_dxlight = abs(Tp[0]*Tp[3]-Tp[1]*Tp[2])
    # ── BSDF product at x1 and x2 ────────────────────────────────────────────
    # Solid-angle-compression factor for refraction through a smooth
    # (delta) dielectric interface -- confirmed against the original SMS
    # paper's reference implementation (specular-manifold-sampling's
    # manifold_ss.cpp specular_reflectance, delta-BSDF branch: `f = 1-F;
    # f *= sqr(eta);`). Verified by hand at normal incidence, where
    # cosHI/cosTM/cosNI all collapse to 1 and the two formulas must agree
    # up to exactly this factor. See
    # project_dielectric_radiance_transmission_bug memory. The eta used is
    # the EMITTER-side orientation (see sms.mojo's sms_walk for the full
    # argument); for this symmetric enter+exit pair eta2 = 1/eta1, so the
    # product is 1 either way and this spelling changes no pixel here --
    # it is the single-refraction chains that actually depend on it.
    var eta1_o = Float32(1)/eta1; var eta2_o = Float32(1)/eta2
    var cosNI1 = abs(dot(n1,wi1)); var cosHI1 = abs(dot(H1,wi1)); var cosTM1 = abs(dot(n1,H1))
    var F1 = fr_dielectric(cosNI1, eta1); var bsdf1 = (Float32(1)-F1)*cosHI1/max(cosNI1*cosTM1*cosTM1,Float32(1e-6)) * eta1_o*eta1_o
    var cosNI2 = abs(dot(n2,wi2)); var cosHI2 = abs(dot(H2,wi2)); var cosTM2 = abs(dot(n2,H2))
    var F2 = fr_dielectric(cosNI2, eta2); var bsdf2 = (Float32(1)-F2)*cosHI2/max(cosNI2*cosTM2*cosTM2,Float32(1e-6)) * eta2_o*eta2_o
    return (True, x1, x2, bsdf1*bsdf2, dx1_dxlight)

@always_inline
# MNEE manifold walk: Newton iteration to find x1 on a glass surface such that
# Snell's law holds for the path x0 → x1 → x2. Works for flat glass surfaces
# (no reprojection needed). Returns (converged, x1_final, det_b, eta_final).
# det_b is the 2x2 constraint Jacobian determinant needed for the transfer matrix.
@always_inline
def _mnee_walk(
    x0: Vec3f,
    x2: Vec3f,
    x1_init: Vec3f,
    n_in: Vec3f,
    dp_du: Vec3f,
    dp_dv: Vec3f,
    eta_in: Float32,
) -> Tuple[Bool, Vec3f, Float32, Float32]:
    var x1 = x1_init
    var n = n_in
    for _iter in range(20):
        var wi3 = x0 - x1
        var wi_len2 = dot(wi3, wi3)
        if wi_len2 < Float32(1e-8):
            return (False, x1_init, Float32(0), eta_in)
        var wi_len = sqrt(wi_len2)
        var wi = wi3 * (Float32(1.0) / wi_len)
        var wo3 = x2 - x1
        var wo_len2 = dot(wo3, wo3)
        if wo_len2 < Float32(1e-8):
            return (False, x1_init, Float32(0), eta_in)
        var wo_len = sqrt(wo_len2)
        var wo = wo3 * (Float32(1.0) / wo_len)
        # eta: pre-computed by caller from dot(pgeo_n_raw, shadow_dir)
        var eta = eta_in
        # Generalized half-vector H = -(wi + eta*wo), Cycles mnee.h convention
        var H3 = -(wi + wo * eta)
        var H_len2 = dot(H3, H3)
        if H_len2 < Float32(1e-10):
            return (False, x1_init, Float32(0), eta)
        var H_len = sqrt(H_len2)
        var H = H3 * (Float32(1.0) / H_len)
        # Tangent frame: s from dp_du projected off n, t = n × s
        var dp_du_dot_n = dot(dp_du, n)
        var s3 = dp_du - n * dp_du_dot_n
        var s_len2 = dot(s3, s3)
        if s_len2 < Float32(1e-10):
            return (False, x1_init, Float32(0), eta)
        var s = s3 * (Float32(1.0) / sqrt(s_len2))
        var tv = cross(n, s)
        # Transmissive check: wi and wo must be on opposite sides of surface
        if dot(n, wi) * dot(n, wo) >= Float32(0.0):
            return (False, x1_init, Float32(0), eta)
        # Constraint: tangential component of H (should be zero at valid refraction)
        var cs = dot(H, s)
        var ct = dot(H, tv)
        # Jacobian b from Cycles mnee_compute_constraint_derivatives (single vertex, flat surface)
        var ilo = eta / (H_len * wo_len)
        var ili = Float32(1.0) / (H_len * wi_len)
        var dH_du = -(dp_du * (ili + ilo)) + wi * (dot(wi, dp_du) * ili) + wo * (dot(wo, dp_du) * ilo)
        var dH_dv = -(dp_dv * (ili + ilo)) + wi * (dot(wi, dp_dv) * ili) + wo * (dot(wo, dp_dv) * ilo)
        dH_du -= H * dot(dH_du, H)
        dH_dv -= H * dot(dH_dv, H)
        dH_du = -dH_du
        dH_dv = -dH_dv
        var b00 = dot(dH_du, s);  var b01 = dot(dH_dv, s)
        var b10 = dot(dH_du, tv); var b11 = dot(dH_dv, tv)
        var det_b = b00 * b11 - b01 * b10
        # Convergence check
        if max(abs(cs), abs(ct)) < SMS_SOLVER_THRESHOLD:
            return (True, x1, det_b, eta)
        # Newton step: solve b * [du, dv] = [cs, ct]
        if abs(det_b) < Float32(1e-5):
            return (False, x1_init, Float32(0), eta)
        var delta_u = (cs * b11 - ct * b01) / det_b
        var delta_v = (-cs * b10 + ct * b00) / det_b
        # Clamp step to prevent divergence
        var step2 = delta_u * delta_u + delta_v * delta_v
        var max_step = wo_len * Float32(0.5)
        if step2 > max_step * max_step:
            var scale = max_step / sqrt(step2)
            delta_u *= scale
            delta_v *= scale
        x1 = x1 - (dp_du * delta_u + dp_dv * delta_v)
    return (False, x1_init, Float32(0), eta_in)

@always_inline
def _nee_infinite_light[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    ilight: InfiniteLight_C,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid,
):
    var env_dir: Vec3f
    var env_rgb: RGB
    var pdf_light: Float32

    if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and _is_real_ptr(ilight.cdf_ptr) and ilight.cdf_w > Int32(0):
        # ── CDF importance sampling, via the shared bvh.mojo implementation ──
        # (A separate "sample a cosine-hemisphere direction, look up the env
        # map there, add it unweighted" block used to live here as an extra
        # "BSDF-sampling" MIS strategy. It's redundant AND wrong: the real
        # BSDF-sampled continuation ray already delivers that same strategy
        # correctly — shade_nee_core's miss handler (below in this file)
        # applies power_heuristic(pdf_bsdf, pdf_light) when a bounce escapes
        # to this same light. The removed block added its full contribution
        # with NO such weight, double-counting unoccluded sky light and
        # washing out shadow contrast — see [[project_object_instancing]] for
        # the barcelona-pavilion/bunny-fur symptom this caused.)
        var (dir_v, rgb_v, pdf_v) = _sample_infinite_light_textured(ilight, Point2f(u_env1, u_env2))
        env_dir = dir_v.to_simd()
        env_rgb = rgb_v
        pdf_light = pdf_v
    else:
        # ── Fallback: cosine-weighted hemisphere (no CDF) ─────────────────
        var _env_s = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), normal)
        env_dir = _env_s[0]
        pdf_light = _env_s[1]
        env_rgb = ilight.scale

    # ── MIS + shadow ray ──────────────────────────────────────────────────
    var cos_env = dot(normal, env_dir)
    if cos_env > Float32(0.0) and not env_rgb.is_black() and pdf_light > Float32(0.0):
        var pdf_bsdf_nee = bxdf_pdf_diffuse(cos_env)
        var mis_w = power_heuristic(pdf_light, pdf_bsdf_nee)
        var contrib = path_ptr[].throughput * bxdf_eval_diffuse(alb) * env_rgb * (cos_env / pdf_light) * mis_w
        var t_max_env = Float32(100000.0)
        _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, env_dir, t_max_env, contrib, guide_write)

@always_inline
def _sample_light_point_and_normal(
    ctx: ShadeContext, al: AreaLight_C, u1: Float32, u2: Float32, mut pcg: PCG32,
) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f]:
    """Point + outward normal on an area light's surface for NEE, plus the
    surface's own (dp_du, dp_dv) tangent basis (used only by
    _nee_area_lights' MNEE glass-refraction focusing — meaningless for a
    curve's round tube, always zero for kind==1 since MNEE is gated off
    there): a random triangle on a mesh light (kind==0, u1/u2 = barycentric
    coords), or a random point on a curve's swept tube (kind==1, u1 =
    position along the picked piece, u2 = angle around the tube). Which
    triangle/piece is picked is its own uniform draw from pcg, independent
    of (u1, u2) — mirrors the pre-existing mesh-light behaviour of picking
    a uniform random triangle rather than area-weighting by triangle."""
    if al.kind == Int8(1):
        var curve = ctx.curves[Int(al.meshIdx)]
        var piece = Int(pcg.next_uint() % UInt32(max(Int(curve.n_pieces), 1)))
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var axis = q1 - q0
        var axis_len = sqrt(dot(axis, axis))
        var axis_dir = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
        if axis_len > Float32(1e-8):
            axis_dir = axis * (Float32(1.0) / axis_len)
        var r = r0 + (r1 - r0) * u1
        var u_perp = _curve_perp_axis(axis_dir)
        var v_perp = cross(axis_dir, u_perp)
        var theta = u2 * TWO_PI
        var radial = u_perp * cos(theta) + v_perp * sin(theta)
        var point = q0 + axis_dir * (axis_len * u1) + radial * r
        var zero3 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (point, radial, zero3, zero3)
    else:
        var lmesh = ctx.meshes[Int(al.meshIdx)]
        var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
        var lb = lti * 3
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = Vec3f(lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = Vec3f(lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = Vec3f(lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
        var sqrt_r1 = sqrt(u1)
        var point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - u2)) + lp2 * (sqrt_r1 * u2)
        var lcross = cross(lp1 - lp0, lp2 - lp0)
        var normal = lcross
        var lcross_len = dot(lcross, lcross)
        if lcross_len > Float32(0.0):
            normal = lcross * (Float32(1.0) / sqrt(lcross_len))
        return (point, normal, lp1 - lp0, lp2 - lp0)

@always_inline
def _sample_area_light_nee(
    ctx: ShadeContext,
    hit_point: Vec3f,
    mut pcg: PCG32,
) -> LightSample:
    """Uniform LightSample wrapper around the mesh/curve area-light picking
    machinery (light_sampler_sample + _sample_light_point_and_normal) — lets
    ANY NEE call site built on the Light/BxDF interface (bvh.mojo's
    LightSample + bxdf.mojo's _nee_weight_simple/_nee_weight_simple_via_spectral/
    _nee_weight_coated_*) consume area lights too, not just the diffuse-
    specific _nee_area_lights MNEE path (which additionally does glass-
    refraction focusing — irrelevant here, this is the plain solid-angle
    pdf every other light type already uses). valid=False when there are no
    area lights or the sampled point is degenerate (zero distance/area, or
    facing away from the shading point)."""
    var invalid = LightSample(Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)),
                               RGB(Float32(0.0)), Float32(0.0), Float32(0.0), False, False)
    if ctx.lights.area_light_count == 0:
        return invalid.copy()
    var u_light = pcg.next_float()
    var ls_result = light_sampler_sample(ctx.lights.light_sampler, u_light)
    var light_idx = ls_result[0]
    var light_sel_pdf = ls_result[1]
    var al = ctx.lights.area_lights[light_idx]
    var r1 = pcg.next_float()
    var r2 = pcg.next_float()
    var (light_point, light_normal, _, _) = _sample_light_point_and_normal(ctx, al, r1, r2, pcg)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    if dist <= Float32(0.0001) or al.total_area <= Float32(0.0):
        return invalid.copy()
    var wi = to_light * (Float32(1.0) / dist)
    var cos_l = -dot(light_normal, wi)
    if cos_l <= Float32(0.0):
        return invalid.copy()
    var pdf = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf <= Float32(0.0):
        return invalid.copy()
    return LightSample(wi, al.emission, pdf, dist, False, True)

@always_inline
@always_inline
def _sms_vertex_from_hit(
    ctx: ShadeContext,
    inter: Intersection_C,
    ray_org: Vec3f,
    shadow_dir: Vec3f,
) -> Tuple[SMSVertex, Bool]:
    """Builds an SMSVertex (flat triangle or curved analytic sphere) from a
    probe intersection along a shadow ray -- the ONE place that decides how
    to turn a glass-surface hit into manifold-walk vertex data, shared by
    all three probe call sites in _sms_probe_and_solve (x1, x2, and the
    N-vertex extra_hits loop) instead of each hand-rolling its own
    triangle-only extraction (that used to be genuinely triangle-only,
    which is why a sphere caustic caster was invisible to SMS/MNEE at all
    -- see project_sms_restir_phase6 memory). Entering-vs-exiting eta
    determination (and, for a triangle, the normal-orientation flip) is
    identical between the two primitive kinds; only how the hit POSITION
    and RAW normal are derived differs. Returns (vertex, ok) -- ok=False
    for a degenerate glass triangle (zero-area) or a degenerate sphere hit
    (exactly at its own center, geometrically impossible but checked for
    safety)."""
    var mat = ctx.materials[Int(inter.primId.materialIndex)]
    var ior = mat.albedo.r
    if inter.primId.type == Int8(4):
        var sph = ctx.lights.spheres[Int(inter.primId.id1)]
        var hit_pt = ray_org + shadow_dir * inter.tHit
        var center = Vec3f(sph.center.x, sph.center.y, sph.center.z)
        var to_hit = hit_pt - center
        var to_hit_len = sqrt(dot(to_hit, to_hit))
        if to_hit_len <= Float32(1e-10):
            return (sms_vertex_init(), False)
        var n_raw = to_hit * (Float32(1.0) / to_hit_len)
        var eta = ior if dot(n_raw, shadow_dir) <= Float32(0.0) else (Float32(1.0) / ior)
        # A normal-mapped caster is a genuinely different manifold: the
        # perturbed normal is what the specular constraint is stated in,
        # and the walk has to re-read the map at every position it visits,
        # so the map travels ON the vertex (see sms.mojo's SMSVertex.nmap).
        # Without this the walk solves the SMOOTH sphere -- one broad root
        # per shading point instead of the many sharp ones a normal map
        # creates, which is the difference between a smooth wash and actual
        # caustic filaments.
        var nmap = normal_slope_map_none()
        if mat.normal_tex_idx >= Int32(0) and _is_real_ptr(ctx.nmaps):
            nmap = ctx.nmaps[Int(mat.normal_tex_idx)]
        return (sms_vertex_sphere(hit_pt, center, sph.radius, eta, nmap, ior), True)
    elif inter.primId.type == Int8(0):
        var (mesh, v0, v1, v2, _) = _get_tri_verts(inter, ctx.meshes)
        var p0 = Vec3f(mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
        var p1 = Vec3f(mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
        var p2 = Vec3f(mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
        var dp_du = p1 - p0
        var dp_dv = p2 - p0
        var n3 = cross(dp_du, dp_dv)
        var n_len = sqrt(dot(n3, n3))
        if n_len <= Float32(1e-10):
            return (sms_vertex_init(), False)
        var n_raw = n3 * (Float32(1.0) / n_len)
        var eta = ior if dot(n_raw, shadow_dir) <= Float32(0.0) else (Float32(1.0) / ior)
        var n = n_raw
        if dot(n, shadow_dir) > Float32(0.0):
            n = -n
        var bu = inter.u; var bv = inter.v
        var point = p0 * (Float32(1.0) - bu - bv) + p1 * bu + p2 * bv
        # Orthonormalize the tangent basis (the reference's
        # ManifoldVertex::make_orthonormal). The triangle's raw edge vectors
        # span the right plane but are neither unit nor perpendicular, so a
        # Jacobian expressed in them is per-PARAMETRIC-unit -- while the
        # estimator pairs it with an AREA-measure light density. Both the
        # specular vertex's basis and the light's have to be orthonormal for
        # |dx1/dxL| to actually be an area-to-area ratio; a sphere vertex
        # already is (its Frisvad frame), which is why only the triangle
        # needed this.
        var (e1, e2) = mnee_orthonormal_basis(dp_du, dp_dv)
        return (sms_vertex_flat(point, n, e1, e2, eta), True)
    else:
        return (sms_vertex_init(), False)

def _sms_probe_glass_chain(
    ctx: ShadeContext,
    start_point: Vec3f,
    shadow_dir: Vec3f,
    start_remaining: Float32,
    max_count: Int,
) -> Tuple[Int, InlineArray[Intersection_C, MAX_SMS_VERTICES], InlineArray[Vec3f, MAX_SMS_VERTICES]]:
    """Probes forward from `start_point` along `shadow_dir` for up to
    `max_count` MORE consecutive dielectric surfaces (Phase 5.1's
    generalization of _mnee_area_light_contribute's own hardcoded
    probe/probe2 pair), stopping at the first non-glass hit or miss.
    Returns (count, hits, origins) -- `hits[0..count)` are valid
    Intersection_C values from ctx's own BVH, same approximate remaining-
    distance bookkeeping the existing probe2 step already uses (each hit's
    tHit is measured from its own segment origin, not the original
    hit_point). `origins[k]` is the ray origin `hits[k].tHit` is measured
    from -- a triangle hit's position can be recovered from barycentrics
    alone, but an analytic sphere hit's cannot (it carries no
    barycentrics), so the caller needs this to reconstruct it via
    origins[k] + shadow_dir*hits[k].tHit."""
    var dummy_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection_C(dummy_prim, Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var hits = InlineArray[Intersection_C, MAX_SMS_VERTICES](fill=dummy_inter)
    var origins = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    var count = 0
    var seg_org = start_point
    var seg_remaining = start_remaining
    for _k in range(max_count):
        var pk_tmax = seg_remaining * Float32(0.9995)
        if pk_tmax <= Float32(0.001):
            break
        var pk_org = seg_org + shadow_dir * Float32(0.0005)
        var pk_ray = Ray_C(Point3f(pk_org[0], pk_org[1], pk_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var pk_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
        traverse_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, pk_ray, pk_tmax, pk_store.unsafe_ptr(),
                           ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                           ctx.lights.spheres, ctx.lights.sphere_count)
        var pk_inter = pk_store[0]
        if pk_inter.hit == Int8(0) or (pk_inter.primId.type != Int8(0) and pk_inter.primId.type != Int8(4)):
            break
        var pk_mat = ctx.materials[Int(pk_inter.primId.materialIndex)]
        if pk_mat.type != MatKind.dielectric and pk_mat.type != MatKind.thin_dielectric:
            break
        hits[count] = pk_inter
        origins[count] = pk_org
        count += 1
        var gk_point: Vec3f
        if pk_inter.primId.type == Int8(4):
            gk_point = pk_org + shadow_dir * pk_inter.tHit
        else:
            var (gk_mesh, gk_v0, gk_v1, gk_v2, _) = _get_tri_verts(pk_inter, ctx.meshes)
            var gk_p0 = Vec3f(gk_mesh.points[gk_v0*4], gk_mesh.points[gk_v0*4+1], gk_mesh.points[gk_v0*4+2])
            var gk_p1 = Vec3f(gk_mesh.points[gk_v1*4], gk_mesh.points[gk_v1*4+1], gk_mesh.points[gk_v1*4+2])
            var gk_p2 = Vec3f(gk_mesh.points[gk_v2*4], gk_mesh.points[gk_v2*4+1], gk_mesh.points[gk_v2*4+2])
            var gk_u = pk_inter.u; var gk_v = pk_inter.v
            gk_point = gk_p0*(Float32(1.0)-gk_u-gk_v) + gk_p1*gk_u + gk_p2*gk_v
        seg_remaining = seg_remaining - pk_inter.tHit
        seg_org = gk_point
    return (count, hits^, origins^)

@always_inline
def _sms_probe_and_solve(
    ctx: ShadeContext,
    hit_point: Vec3f,
    shadow_dir: Vec3f,
    dist: Float32,
    light_point: Vec3f,
    ldp_du_v: Vec3f,
    ldp_dv_v: Vec3f,
    mut pcg: PCG32,
) -> Tuple[Bool, Bool, Int, InlineArray[SMSVertex, MAX_SMS_VERTICES], Float32, Float32, Float32]:
    """Probe for up to MAX_SMS_VERTICES glass surfaces between `hit_point`
    and `light_point` and, if found, solve the resulting specular chain --
    pure geometry/Newton-solve logic shared by `_mnee_area_light_contribute`
    (direct-contribution plain-NEE/ReSTIR-DI path) and ReSTIR SMS's
    candidate generation, factored out so both draw from ONE implementation
    of the probing+dispatch logic instead of two copies drifting apart.
    No access to path_ptr/alb/emission/lobe_w -- those are caller-specific.

    Returns (dielectric_found, solve_ok, n_vertices, verts, bsdf_product,
    dx1_dxlight, trials). `dielectric_found` alone governs whether a
    caller should skip its own straight shadow ray (the light IS occluded
    by glass either way); `solve_ok` governs whether there's an actual
    refracted contribution to use -- mirrors the original function
    returning True (skip shadow ray) even when the manifold walk itself
    failed to converge. `verts[i].pos` holds the SOLVED positions on
    success; `verts[i].normal/dp_du/dp_dv/eta` are always the probed
    triangle's own fixed properties, valid even when `solve_ok=False`
    (a caller doing manifold-shift reuse needs the probed vertex 0's
    geometry regardless of whether the direct-probe seed itself
    converged). `trials` is always 1.0 for the 1-/2-vertex MNEE fast path
    (Phase 5.4, no Bernoulli-trial estimator there) and
    sms_solve_bernoulli's own trial count for N>=3 chains (Phase 5.3)."""
    var zero_verts = InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    # MNEE: probe for up to 2 glass surfaces between hit_point and light.
    # For each probe hit we detect entering/exiting from dot(n_raw, probe_dir).
    var probe_org = hit_point + shadow_dir * Float32(0.0002)
    var probe_ray = Ray_C(
        Point3f(probe_org[0], probe_org[1], probe_org[2]),
        Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection_C(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
    traverse_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                       ctx.lights.spheres, ctx.lights.sphere_count)
    var probe_inter = probe_store[0]
    if probe_inter.hit == Int8(0) or (probe_inter.primId.type != Int8(0) and probe_inter.primId.type != Int8(4)):
        return (False, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var probe_mat = ctx.materials[Int(probe_inter.primId.materialIndex)]
    if probe_mat.type != MatKind.dielectric and probe_mat.type != MatKind.thin_dielectric:
        return (False, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))

    # --- Extract x1 geometry (triangle or analytic sphere) ---
    var (v1, v1_ok) = _sms_vertex_from_hit(ctx, probe_inter, probe_org, shadow_dir)
    if not v1_ok:
        # Degenerate glass surface: a dielectric IS in the way (so the
        # straight shadow ray is blocked regardless) but no usable manifold
        # geometry.
        return (True, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var is_sphere1 = v1.is_sphere != Int8(0)
    var pgeo_n = v1.normal
    var pdp_du = v1.dp_du
    var pdp_dv = v1.dp_dv
    var eta1 = v1.eta
    var x1_init = v1.pos
    # --- Probe for second glass surface beyond x1 ---
    # Deliberately SKIPPED for a sphere caster: read directly from the real
    # SMS reference renderer's own source (manifold_ss.cpp,
    # SpecularManifoldSingleScatter::newton_solver/sample_path) to root-
    # cause why gonzales's own real-scene render of sphere_sms.xml wasn't
    # converging -- their "single scatter" integrator solves for exactly
    # ONE specular vertex per caustic-caster shape (one refraction event),
    # full stop; it never chains an entry+exit pair through a solid
    # object. Building that harder 2-vertex problem (this file did, before
    # this fix) is solving something genuinely harder than what the
    # reference scene's own reference image was ever generated from: on
    # real (non-toy) sphere geometry the coupled entry+exit Newton solve
    # kept converging to a root literally behind the sphere, embedded
    # under the scene's own floor -- physically correct rejection once an
    # occlusion-aware reprojection was added (see sms.mojo's
    # `_sms_reproject_onto_sphere_anchored`), but then unable to re-
    # converge to a valid PAIR at all, because "entry connects to exit
    # without crossing the floor" isn't a constraint the per-vertex
    # tangential Newton solve is even solving for. Matching the reference's
    # own scope avoids the problem instead of fighting it: treat a sphere
    # caustic caster as a single idealized refraction (like a thin shell),
    # routing it through the 1-vertex `sms_walk` path below -- already
    # validated correct and convergent (project_sms_restir_phase6 memory).
    # A flat 2-surface pane (an actual window, unrelated to this) is
    # untouched -- `is_sphere1` only ever true for `Sphere_C`.
    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001) and not is_sphere1:
        var probe2_ray = Ray_C(
            Point3f(probe2_org[0], probe2_org[1], probe2_org[2]),
            Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
        traverse_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                   ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                   ctx.lights.spheres, ctx.lights.sphere_count)
        probe2_inter = probe2_store[0]
    var has_second_glass = False
    if probe2_inter.hit != Int8(0) and (probe2_inter.primId.type == Int8(0) or probe2_inter.primId.type == Int8(4)):
        var probe2_mat_c = ctx.materials[Int(probe2_inter.primId.materialIndex)]
        has_second_glass = (probe2_mat_c.type == MatKind.dielectric or probe2_mat_c.type == MatKind.thin_dielectric)

    if not has_second_glass:
        var verts1 = InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
        verts1[0] = v1
        if is_sphere1:
            # A curved caster needs sms_walk's general (curvature-aware)
            # Newton solve -- _mnee_walk assumes a flat tangent plane IS the
            # surface, which only holds for a triangle -- and it needs the
            # FULL SMS estimator around that solve rather than one
            # deterministic probe-seeded shot.
            #
            # MNEE's single-seed shortcut is justified by uniqueness: for a
            # flat refractor the solution given a probe is essentially the
            # only one, so finding it once is finding all of it. A sphere
            # already strains that, and a NORMAL-MAPPED sphere breaks it
            # outright -- the perturbed surface has many specular solutions
            # per shading point, and they are exactly what the caustic's
            # filament structure is made of. One deterministic seed reports
            # a single root with weight 1 and drops the rest, which renders
            # as the right pattern at a fraction of the right brightness.
            # sms_solve_bernoulli draws its seeds at random over the sphere
            # instead and weights by the reciprocal probability of
            # rediscovering the root it found (Zeltner et al. 2020), which
            # is an unbiased estimate of the sum over all of them. The
            # reflection/validity rejection the reference applies after each
            # solve now lives inside sms_walk itself, so it covers every
            # trial rather than just this call site's own.
            var _bern1 = sms_solve_bernoulli(
                hit_point, light_point, verts1, 1, ldp_du_v, ldp_dv_v,
                Float32(0.0), pcg,
                ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                ctx.lights.spheres, ctx.lights.sphere_count)
            if not _bern1[0]:
                return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
            verts1[0].pos = _bern1[1][0]
            sms_refresh_solved_frames(verts1, 1)
            return (True, True, 1, verts1.copy(), _bern1[2], _bern1[3], _bern1[4])
        # --- 1-vertex MNEE fast path (flat triangle only, unchanged) ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(
            hit_point, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok:
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var wi_f = hit_point - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1.0) / wi_len_f)
        var wo_fn = wo_f * (Float32(1.0) / wo_len_f)
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1.0) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var s_f = s3_f * (Float32(1.0) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du_v - wo_fn * dot(wo_fn, ldp_du_v)) * ilo_l
        var dHdv_l = (ldp_dv_v - wo_fn * dot(wo_fn, ldp_dv_v)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1.0) - F_r
        # 1/(eta*eta): see sms.mojo's sms_walk for why the compression
        # factor takes the EMITTER-side orientation of eta while `eta_f`
        # carries the constraint-side one. Unlike the 2-vertex slab above,
        # a single refraction has nothing to cancel against, so getting
        # this backwards here really is an eta^4 error.
        var eta_f_o = Float32(1.0) / eta_f
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6)) * eta_f_o*eta_f_o
        verts1[0].pos = x1_f
        return (True, True, 1, verts1.copy(), bsdf_s, dx1_dxl, Float32(1.0))

    var (v2, v2_ok) = _sms_vertex_from_hit(ctx, probe2_inter, probe2_org, shadow_dir)
    if not v2_ok:
        return (True, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var is_sphere2 = v2.is_sphere != Int8(0)
    var pgeo_n2 = v2.normal
    var pdp_du2 = v2.dp_du
    var pdp_dv2 = v2.dp_dv
    var eta2 = v2.eta
    var x2_init = v2.pos

    # ── Phase 5.1: probe past x2 for a 3rd+ glass surface. Only entered
    # when the chain is genuinely longer than MNEE's own 1-/2-vertex scope
    # (5.4) -- an ordinary 2-surface pane (the overwhelming majority of
    # real glass) never reaches this, so this branch adds no cost there.
    var _chain_res = _sms_probe_glass_chain(
        ctx, x2_init, shadow_dir, dist - probe2_inter.tHit - probe2_t0,
        MAX_SMS_VERTICES - 2)
    var extra_count = _chain_res[0]
    var extra_hits = _chain_res[1].copy()
    var extra_origins = _chain_res[2].copy()

    if extra_count == 0 and not is_sphere1 and not is_sphere2:
        # --- 2-vertex MNEE (unchanged fast path, Phase 5.4) ---
        var verts2 = InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
        verts2[0] = sms_vertex_flat(x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        verts2[1] = sms_vertex_flat(x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2)
        var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
            hit_point, light_point,
            x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
            x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
            ldp_du_v, ldp_dv_v)
        if not ok2:
            return (True, False, 2, verts2.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        verts2[0].pos = x1_f2
        verts2[1].pos = x2_f2
        return (True, True, 2, verts2.copy(), bsdf_prod, dx1_dxl2, Float32(1.0))

    if extra_count == 0:
        # --- 2-vertex chain with at least one curved (sphere) caster:
        # route through sms_walk's general, curvature-aware solve instead
        # of the flat-only _mnee_walk2 fast path. ---
        var verts2c = InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
        verts2c[0] = v1
        verts2c[1] = v2
        var _walk2c = sms_walk(hit_point, light_point, verts2c.copy(), 2, ldp_du_v, ldp_dv_v,
            ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
            ctx.lights.spheres, ctx.lights.sphere_count)
        var ok2c = _walk2c[0]
        var pos2c = _walk2c[1].copy()
        var bsdf2c = _walk2c[2]
        var jac2c = _walk2c[3]
        # A sphere (unlike a flat pane, where the fast path's single Newton
        # solve is provably the unique stationary point) can have more than
        # one mathematically valid specular chain -- the straight-line probe
        # seed sits in the basin of attraction of whichever root is nearest,
        # but on real (non-toy) geometry that solve can wander all the way
        # to the FAR hemisphere of the sphere: a root that satisfies the
        # local Snell's-law constraint just as well, but is physically
        # unreachable (behind the sphere from the shading point's own
        # perspective, often literally embedded in another surface like
        # this scene's floor). Found via real-scene diagnostics on
        # sphere_sms.xml: `_sms_probe_and_solve` reported solve_ok=True with
        # plausible-looking bsdf_product/jacobian values on ~100% of its
        # candidates, yet `sms_target_pdf`'s cos_s_x0 gate rejected every
        # single one -- the solved entry vertex had landed on the sphere's
        # underside (y around -5.5 on a radius-6.5 sphere resting on a
        # y=0 floor), nowhere near the seed. Reject here instead of relying
        # solely on the caller's downstream visibility gate: cheaply checks
        # that each curved vertex didn't cross to the opposite hemisphere
        # from where the probe originally hit it (`cos_s_x0`'s check only
        # covers x0's own normal, not whether the SPHERE root itself is the
        # near one) -- generous 60-degree threshold, well past any
        # legitimate MNEE deflection for a realistic IOR, tight enough to
        # catch a same-scene wrong-hemisphere jump (which measured over 90
        # degrees).
        if ok2c and verts2c[0].is_sphere != Int8(0):
            var seed0 = verts2c[0].pos - verts2c[0].sphere_center
            var sol0 = pos2c[0] - verts2c[0].sphere_center
            var denom0 = sqrt(dot(seed0, seed0)) * sqrt(dot(sol0, sol0))
            if denom0 <= Float32(1e-12) or dot(seed0, sol0) / denom0 < Float32(0.5):
                ok2c = False
        if ok2c and verts2c[1].is_sphere != Int8(0):
            var seed1 = verts2c[1].pos - verts2c[1].sphere_center
            var sol1 = pos2c[1] - verts2c[1].sphere_center
            var denom1 = sqrt(dot(seed1, seed1)) * sqrt(dot(sol1, sol1))
            if denom1 <= Float32(1e-12) or dot(seed1, sol1) / denom1 < Float32(0.5):
                ok2c = False
        if not ok2c:
            return (True, False, 2, verts2c.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        verts2c[0].pos = pos2c[0]
        verts2c[1].pos = pos2c[1]
        sms_refresh_solved_frames(verts2c, 2)
        return (True, True, 2, verts2c.copy(), bsdf2c, jac2c, Float32(1.0))

    # --- N-vertex SMS (Phase 5.1/5.2/5.3, sms.mojo) ---
    var n_total = 2 + extra_count
    var verts = InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    verts[0] = v1
    verts[1] = v2
    var chain_ok = True
    for k in range(extra_count):
        var (ek_vert, ek_ok) = _sms_vertex_from_hit(ctx, extra_hits[k], extra_origins[k], shadow_dir)
        if not ek_ok:
            chain_ok = False; break
        verts[2+k] = ek_vert
    if not chain_ok:
        return (True, False, n_total, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    # Jitter scale (5.2): a small fraction of the average spacing between
    # adjacent probed vertices -- keeps the re-seed within the flat-
    # triangle validity radius each vertex's own straight-line probe
    # already established.
    var span = light_point - hit_point
    var span_len = sqrt(dot(span, span))
    var jitter_scale = Float32(0.05) * (span_len / Float32(max(n_total, 1)))
    var _bern_res = sms_solve_bernoulli(
        hit_point, light_point, verts, n_total, ldp_du_v, ldp_dv_v, jitter_scale, pcg,
        ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
        ctx.lights.spheres, ctx.lights.sphere_count)
    var sms_ok = _bern_res[0]
    var sms_pos = _bern_res[1].copy()
    var sms_bsdf = _bern_res[2]
    var sms_jac = _bern_res[3]
    var sms_trials = _bern_res[4]
    if not sms_ok:
        return (True, False, n_total, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    for i in range(n_total):
        verts[i].pos = sms_pos[i]
    sms_refresh_solved_frames(verts, n_total)
    return (True, True, n_total, verts.copy(), sms_bsdf, sms_jac, sms_trials)

@always_inline
def _mnee_area_light_contribute(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    shadow_dir: Vec3f,
    dist: Float32,
    light_point: Vec3f,
    ldp_du_v: Vec3f,
    ldp_dv_v: Vec3f,
    al: AreaLight_C,
    inv_pdf_area: Float32,
    lobe_w: Float32,
) -> Bool:
    """MNEE glass-caustic probing for one already-chosen area-light sample,
    factored out of `_nee_area_lights` so ReSTIR's `di_resolve` can use the
    identical transport (it previously could not, which lost ALL light
    reaching a surface through a dielectric on the ReSTIR path -- ~9% of
    staircase2, concentrated behind its glass railing).

    Probes for up to MAX_SMS_VERTICES glass surfaces between `hit_point` and
    `light_point`; on a hit, runs the 1-/2-vertex MNEE fast path (Phase
    5.4) or, for a genuinely longer chain, sms.mojo's general N-vertex
    manifold walk with random seeding and the Bernoulli-trial reciprocal
    estimator (Phase 5.1-5.3), adds the refracted contribution directly to
    `path_ptr[].estimate`, and returns True. The caller MUST then skip its
    own straight shadow ray -- MNEE/SMS REPLACES it
    (that ray is occluded by the glass by construction, so tracing it would
    contribute nothing anyway, but the `used_mnee` bookkeeping is what keeps
    the two strategies from being double-counted if that ever stops holding).
    Returns False when no dielectric intervenes, leaving the caller to do
    ordinary shadow-ray NEE.

    `inv_pdf_area` is 1/p_A(y), the reciprocal AREA-measure density of the
    light sample -- the one quantity that legitimately differs between the
    two callers, hence a parameter rather than a recomputation:
      * plain NEE  passes `al.total_area / light_sel_pdf` (the sample really
        was drawn with that area density);
      * ReSTIR DI  passes `W * dist^2 / cos_l`, converting its RIS weight
        (which estimates 1/q in SOLID-ANGLE measure, since
        `_di_sample_candidate`'s gen_pdf carries the dist^2/cos_l factor)
        back into area measure via q_sa = q_A * dist^2/cos_l. Getting this
        conversion wrong is silent: it only shows up as a scene-dependent
        brightness error on glass, exactly the class of bug this whole
        area is prone to.

    No MIS weight is applied here, matching the pre-existing plain-NEE
    behaviour: the refracted path is a specular chain that BSDF sampling
    effectively never reproduces, so there is no competing strategy to
    balance against."""
    # MNEE's glass-refraction focusing needs the light's own surface
    # tangents (ldp_du_v/ldp_dv_v) — well-defined for a flat mesh triangle,
    # not for a curve's swept tube. Curve lights (kind==1) fall back to
    # plain (non-MNEE) shadow-ray NEE through glass.
    if al.kind != Int8(0):
        return False

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    # ORTHONORMALIZE the light's tangent basis before handing it to the
    # manifold walk. `_sample_light_point_and_normal` returns raw triangle
    # EDGES (lp1-lp0, lp2-lp0), which are neither unit-length nor
    # perpendicular, so d(constraint)/d(light) comes out in that triangle's
    # own parametric units -- while `inv_pdf_area` is an AREA-measure
    # density. The two only pair up when the light basis is orthonormal, and
    # the reference makes exactly this call (`vy.make_orthonormal()` on the
    # emitter vertex inside geometric_term) for the same reason.
    var (l_du, l_dv) = mnee_orthonormal_basis(ldp_du_v, ldp_dv_v)
    var _probe_res = _sms_probe_and_solve(
        ctx, hit_point, shadow_dir, dist, light_point, l_du, l_dv, pcg)
    var dielectric_found = _probe_res[0]
    var solve_ok = _probe_res[1]
    var n = _probe_res[2]
    var verts = _probe_res[3].copy()
    var bsdf_product = _probe_res[4]
    var dx1_dxlight = _probe_res[5]
    var trials = _probe_res[6]
    path_ptr[].pcgState = pcg.state
    if not dielectric_found:
        return False
    if not solve_ok:
        return True

    # Unified "use the solved chain" step -- one formula regardless of
    # whether it came from the 1-/2-vertex MNEE fast path or sms.mojo's
    # general N-vertex walk (Phase 5.1's own module docstring explains why
    # the two are physically the same quantity, just chain-length-general).
    var first_vertex = verts[0].pos
    # GEOMETRIC normal, matching the reference's own geometric term
    # (`dw0_dx1 = abs_dot(d, v1.gn) * inv_r2` -- v1.gn, not the shading
    # normal). They differ on a normal-mapped caster, where `normal` carries
    # the map's perturbation; a solid-angle-to-area conversion is a property
    # of the SURFACE, not of the normal the BSDF is shaded with.
    var first_normal = verts[0].normal
    if verts[0].is_sphere != Int8(0):
        var gnv = verts[0].pos - verts[0].sphere_center
        var gnl = sqrt(dot(gnv, gnv))
        if gnl > Float32(1e-8):
            first_normal = gnv * (Float32(1.0) / gnl)
    var last_vertex = verts[n-1].pos
    var wi_f = hit_point - first_vertex
    var wi_len = sqrt(dot(wi_f, wi_f))
    if wi_len <= Float32(1e-8):
        return True
    var wi_fn = wi_f * (Float32(1.0) / wi_len)
    var cos_s_x0 = dot(normal, -wi_fn)
    if cos_s_x0 <= Float32(0.0):
        return True
    var dw0_dx1 = abs(dot(wi_fn, first_normal)) / (wi_len * wi_len)
    var g = min(dw0_dx1 * dx1_dxlight, Float32(2.0))
    var wo_f = light_point - last_vertex
    var wo_len = sqrt(dot(wo_f, wo_f))
    if wo_len <= Float32(1e-8):
        return True
    var wo_fn = wo_f * (Float32(1.0) / wo_len)
    var vis_org = last_vertex + wo_fn * Float32(0.001)
    var vis_ray = Ray_C(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
    # A sphere caustic caster's outgoing leg necessarily re-crosses its OWN
    # far side (the unmodeled exit refraction of the single-vertex model,
    # see any_hit_bvh2_core's own docstring) -- exclude just that one
    # sphere from this occlusion test, not a real distinct occluder.
    var ign_center = Vec3f(Float32(0.0))
    var ign_radius = Float32(-1.0)
    if verts[n-1].is_sphere != Int8(0):
        ign_center = verts[n-1].sphere_center
        ign_radius = verts[n-1].sphere_radius
    if any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis_ray, wo_len * Float32(0.999),
                          ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                          ctx.lights.spheres, ctx.lights.sphere_count,
                          ign_center, ign_radius):
        return True
    # Phase 5.5: no MIS weight -- the refracted path is a specular chain
    # that BSDF sampling effectively never reproduces, so there is no
    # competing strategy to balance against. `trials` is exactly 1.0 for
    # the 1-/2-vertex fast path (no Bernoulli-trial estimator there) so
    # this reduces to the original formula unchanged in that case.
    var mnee_wt = bxdf_eval_diffuse(alb) * al.emission * (cos_s_x0 * g * bsdf_product * lobe_w * inv_pdf_area * trials)
    path_ptr[].estimate += path_ptr[].throughput * mnee_wt
    return True

# ── ReSTIR SMS (Phase 6's remaining first piece: candidate generation) ──────
# Reuses _sms_probe_and_solve for the identical probe+Newton-solve logic
# _mnee_area_light_contribute's direct-contribution path already uses --
# see that function's docstring and project_sms_restir_phase6 memory for
# why this is a SEPARATE function rather than a third calling convention
# threaded through the already-validated direct-contribution one.
#
# Unlike ReSTIR DI's di_generate_reservoir (M=16 cheap light-sampler
# draws streamed via RIS), SMS has effectively M=1 per pixel per frame --
# generating even ONE admissible chain is the expensive part (a full
# Newton solve, possibly Bernoulli trials for N>=3). ReSTIR's value here
# is almost entirely about reusing that one expensive solve across frames
# (temporal) and pixels (spatial) via sms_shift, not about resampling
# among many cheap candidates. Unlike ReSTIR GI's candidate generation
# (split across two bounces, target pdf deferred to a later call because
# it needs data unavailable until then), SMS has every quantity
# sms_target_pdf needs available immediately at generation time -- so,
# like di_generate_reservoir, this returns an ALREADY-STREAMED reservoir,
# not an unstreamed candidate.
def sms_generate_reservoir(
    ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    shadow_dir: Vec3f, dist: Float32, light_point: Vec3f,
    ldp_du_v: Vec3f, ldp_dv_v: Vec3f,
    al: AreaLight_C, inv_pdf_area: Float32, mut pcg: PCG32,
) -> Tuple[Bool, SMSReservoir]:
    """Phase 6's candidate generation: probe+solve for an admissible
    specular chain toward the given area-light sample (same scope
    restriction as _mnee_area_light_contribute -- mesh lights only, kind
    0; curve lights fall back to ordinary shadow-ray NEE, no SMS
    candidate generated) and, on success, stream it into a fresh
    reservoir with RIS weight p̂/q where p̂ is sms_target_pdf and q folds
    in the light's own area-measure generation density (`inv_pdf_area`
    is 1/q, matching _mnee_area_light_contribute's own parameter of the
    same name) and the Bernoulli-trial reciprocal estimator (`trials`,
    1.0 for the 1-/2-vertex fast path).

    Returns (dielectric_found, reservoir) -- mirrors
    _sms_probe_and_solve's own split: `dielectric_found` alone tells the
    caller whether to skip its own straight shadow ray (the light IS
    occluded by glass either way), separate from whether the reservoir
    actually holds a candidate (n_vertices=0 covers "no dielectric",
    "solve failed to converge", and "target function evaluates to zero"
    identically -- all "no candidate this frame", matching
    di_generate_reservoir's own zero-weight handling). Conflating the two
    would be a real bug: reporting `dielectric_found=True` whenever
    `al.kind==0` (i.e. for every mesh light, glass or not) would make a
    temporal-reuse caller skip ordinary shadow-ray NEE for every mesh
    light in every scene, not just ones with glass in the way."""
    var res = sms_reservoir_init()
    if al.kind != Int8(0):
        return (False, res^)
    var _probe_res2 = _sms_probe_and_solve(
        ctx, hit_point, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, pcg)
    var dielectric_found = _probe_res2[0]
    var solve_ok = _probe_res2[1]
    var n = _probe_res2[2]
    var verts = _probe_res2[3].copy()
    var bsdf_product = _probe_res2[4]
    var dx1_dxlight = _probe_res2[5]
    var trials = _probe_res2[6]
    if not dielectric_found or not solve_ok:
        return (dielectric_found, res^)
    var p_hat = sms_target_pdf(hit_point, normal, alb, verts[0].pos, verts[0].normal, al.emission, bsdf_product, dx1_dxlight)
    var weight = p_hat * inv_pdf_area * trials
    var accept = reservoir_update(res.state, weight, pcg.next_float())
    if accept:
        res.n_vertices = Int32(n)
        res.verts = verts.copy()
        res.light_point = light_point
        res.ldp_du = ldp_du_v
        res.ldp_dv = ldp_dv_v
        res.le = al.emission
        res.bsdf_product = bsdf_product
        res.dx1_dxlight = dx1_dxlight
    return (dielectric_found, res^)

# Defensive cap on a finalized SMS reservoir's own state.w -- same fix,
# same root cause, as DI_MAX_FINALIZED_WEIGHT/GI_MAX_FINALIZED_WEIGHT
# (see DI_MAX_FINALIZED_WEIGHT's own comment for the full mechanism):
# reservoir_combine's weight formula multiplies in a source reservoir's
# own already-finalized state.w, so one anomalous value compounds into
# unbounded growth across many frames of temporal reuse without a bound.
# Applied proactively here, not discovered via a live bug this time --
# both DI and GI needed this fix independently before it was understood
# to be a structural property of reservoir_combine itself, not a
# payload-specific quirk (see project_restir_migration.md memory).
comptime SMS_MAX_FINALIZED_WEIGHT: Float32 = Float32(10.0)
comptime SMS_TEMPORAL_M_CAP: Float32 = Float32(64.0)

def sms_resolve(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut res: SMSReservoir,
):
    """Finalizes the reservoir's RIS weight (Bitterli et al. 2020 Eq. 6,
    via reservoir_finalize), then traces one shadow ray for the winning
    chain's last vertex -> light segment and adds its weighted
    contribution -- mirrors di_resolve/_mnee_area_light_contribute's own
    "use the result" step (same wi/cos_s_x0/visibility-ray pattern), the
    one piece sms_target_pdf itself doesn't cover since it only computes
    the scalar magnitude, not the per-channel Le-weighted contribution
    or the occlusion test. No MIS weight, same reasoning as
    _mnee_area_light_contribute's own (module docstring there): an SMS
    chain is a specular path BSDF sampling effectively never reproduces."""
    if res.n_vertices <= Int32(0):
        return
    var p_hat = sms_target_pdf(hit_point, normal, alb, res.verts[0].pos, res.verts[0].normal, res.le, res.bsdf_product, res.dx1_dxlight)
    reservoir_finalize(res.state, p_hat)
    if res.state.w > SMS_MAX_FINALIZED_WEIGHT:
        res.state.w = SMS_MAX_FINALIZED_WEIGHT
    if res.state.w <= Float32(0.0):
        return
    var n = Int(res.n_vertices)
    var first_vertex = res.verts[0].pos
    var last_vertex = res.verts[n - 1].pos
    var wi_f = hit_point - first_vertex
    var wi_len = sqrt(dot(wi_f, wi_f))
    if wi_len <= Float32(1e-8):
        return
    var wi_fn = wi_f * (Float32(1.0) / wi_len)
    var cos_s_x0 = dot(normal, -wi_fn)
    if cos_s_x0 <= Float32(0.0):
        return
    var wo_f = res.light_point - last_vertex
    var wo_len = sqrt(dot(wo_f, wo_f))
    if wo_len <= Float32(1e-8):
        return
    var wo_fn = wo_f * (Float32(1.0) / wo_len)
    var vis_org = last_vertex + wo_fn * Float32(0.001)
    var vis_ray = Ray_C(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
    # See _mnee_area_light_contribute's identical fix: a sphere caustic
    # caster's outgoing leg necessarily re-crosses its OWN far side, so
    # exclude just that one sphere from this occlusion test.
    var ign_center = Vec3f(Float32(0.0))
    var ign_radius = Float32(-1.0)
    if res.verts[n-1].is_sphere != Int8(0):
        ign_center = res.verts[n-1].sphere_center
        ign_radius = res.verts[n-1].sphere_radius
    if any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis_ray, wo_len * Float32(0.999),
                          ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                          ctx.lights.spheres, ctx.lights.sphere_count,
                          ign_center, ign_radius):
        return
    var contrib = bxdf_eval_diffuse(alb) * res.le * (cos_s_x0 * res.state.w)
    path_ptr[].estimate += path_ptr[].throughput * contrib

def sms_temporal_step(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    shadow_dir: Vec3f, dist: Float32, light_point: Vec3f,
    ldp_du_v: Vec3f, ldp_dv_v: Vec3f,
    al: AreaLight_C, inv_pdf_area: Float32, mut pcg: PCG32,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    pixel_idx: Int = -1,
) -> Bool:
    """Phase 6's temporal-only driver (no spatial reuse yet -- see
    project_sms_restir_phase6.md memory for the remaining scope):
    generate this frame's fresh candidate (sms_generate_reservoir),
    temporally combine with the previous frame's stored reservoir at the
    SAME pixel via IDENTITY reprojection (gonzales has no motion support,
    so x0/light_point are unchanged frame to frame -- no shift/Jacobian
    needed, matching di_temporal_step's own reasoning for the DI case),
    resolve one shadow ray for the combined winner, then M-cap and store
    for the next frame. Returns True iff a dielectric was found this
    frame (mirrors _mnee_area_light_contribute's own return convention --
    the caller must skip its ordinary shadow ray in that case, regardless
    of whether resolve produced a visible contribution). Falls back to
    plain single-frame candidate generation + resolve (no persistence)
    when sms_io isn't real -- matches di_temporal_step's own fallback for
    the batch (non-interactive) path."""
    var gen_result = sms_generate_reservoir(ctx, hit_point, normal, alb, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, al, inv_pdf_area, pcg)
    var dielectric_found = gen_result[0]
    var res = gen_result[1].copy()
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(sms_io.read)
    if has_temporal:
        var prev = sms_io.read[pixel_idx].copy()
        if prev.n_vertices > Int32(0):
            dielectric_found = True
            var p_hat_prev_here = sms_target_pdf(hit_point, normal, alb, prev.verts[0].pos, prev.verts[0].normal, prev.le, prev.bsdf_product, prev.dx1_dxlight)
            var accept = reservoir_combine(res.state, prev.state, p_hat_prev_here, pcg.next_float())
            if accept:
                res.n_vertices = prev.n_vertices
                res.verts = prev.verts.copy()
                res.light_point = prev.light_point
                res.ldp_du = prev.ldp_du
                res.ldp_dv = prev.ldp_dv
                res.le = prev.le
                res.bsdf_product = prev.bsdf_product
                res.dx1_dxlight = prev.dx1_dxlight

    sms_resolve(path_ptr, ctx, hit_point, normal, alb, res)
    if has_temporal:
        reservoir_cap_confidence(res.state, SMS_TEMPORAL_M_CAP)
        sms_io.write[pixel_idx] = res^
    return dielectric_found

def _nee_area_lights[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    u_light: Float32,
    u_bary1: Float32,
    u_bary2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid,
    # Selection-probability compensation for compound BxDFs with more than one
    # lobe (e.g. diffuse_transmit's reflect/transmit split) — see bxdf_sample_
    # diffuse_transmit. 1.0 for a plain single-lobe Lambertian surface.
    lobe_w: Float32 = Float32(1.0),
    # Phase 6 (ReSTIR SMS): real only from _shade_diffuse_nee's bounce-0
    # call site when --sms-restir is active -- every other call site
    # (diffuse_transmit's own NEE, deeper bounces) leaves this at the
    # dangling default, getting exactly today's plain-MNEE-every-bounce
    # behavior. sms_temporal_step has no lobe_w parameter (assumes 1.0,
    # single-lobe diffuse only) -- fine today since no compound-BxDF call
    # site ever passes a real sms_io, but would need extending before one
    # could.
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    pixel_idx: Int = -1,
):
    if ctx.lights.area_light_count == 0:
        return
    var ls_u_nee = u_light
    var ls_result_nee = light_sampler_sample(ctx.lights.light_sampler, ls_u_nee)
    var light_idx = ls_result_nee[0]
    var light_sel_pdf_nee = ls_result_nee[1]
    var al = ctx.lights.area_lights[light_idx]
    var (light_point, light_normal, ldp_du_v, ldp_dv_v) = _sample_light_point_and_normal(ctx, al, u_bary1, u_bary2, pcg)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    if dist > Float32(0.0001) and al.total_area > Float32(0.0):
        var shadow_dir = to_light * (Float32(1.0) / dist)
        var cos_s = dot(normal, shadow_dir)
        var cos_l = -dot(light_normal, shadow_dir)
        if cos_s > Float32(0.0) and cos_l > Float32(0.0):
            var pdf_light = dist_sq * light_sel_pdf_nee / (cos_l * al.total_area)
            var pdf_bsdf_nee = bxdf_pdf_diffuse(cos_s)
            var w_nee = power_heuristic(pdf_light, pdf_bsdf_nee)
            var weight = bxdf_eval_diffuse(alb) * al.emission * (cos_s * w_nee * lobe_w / pdf_light)
            var contrib = path_ptr[].throughput * weight

            if pixel_idx >= 0 and _is_real_ptr(sms_io.read):
                # ReSTIR SMS (Phase 6): temporal-reused glass-caustic probing
                # in place of plain per-frame MNEE. Same used/skip-shadow-ray
                # convention as the plain-MNEE branch below.
                var used_sms = sms_temporal_step(
                    path_ptr, ctx, hit_point, normal, alb, shadow_dir, dist,
                    light_point, ldp_du_v, ldp_dv_v, al,
                    al.total_area / light_sel_pdf_nee, pcg, sms_io, pixel_idx)
                if not used_sms:
                    _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, shadow_dir, dist * Float32(0.9999), contrib, guide_write)
            else:
                # MNEE glass-caustic probing, shared verbatim with ReSTIR DI's
                # di_resolve via _mnee_area_light_contribute above. Returns True
                # when a dielectric intervened and it has already added the
                # refracted contribution -- in that case the straight shadow ray
                # below is deliberately skipped (MNEE replaces it, it does not
                # supplement it).
                # Record whether MNEE/SMS took over the direct lighting here,
                # so a later BSDF-sampled arrival at the emitter through a
                # specular chain -- the SAME path family this strategy just
                # sampled -- is not counted a second time. See
                # PathState_C.sms_covered.
                var used_mnee = _mnee_area_light_contribute(
                    path_ptr, ctx, normal, hit_point, alb, shadow_dir, dist,
                    light_point, ldp_du_v, ldp_dv_v, al,
                    al.total_area / light_sel_pdf_nee, lobe_w)
                # Record the vertex, not the outcome. Whether MNEE covers a
                # given emitter point is a property of that POINT (is there
                # glass on the segment?), not of which light this sample
                # happened to pick -- so the decision is deferred to the
                # emitter-hit site, which knows the point actually reached.
                # Keying it on `used_mnee` instead makes the suppression
                # depend on the NEE light draw, which is independent of where
                # the BSDF ray goes, and that IS biased whenever a vertex has
                # glass toward some lights and not others.
                path_ptr[].sms_covered = Int8(1)
                path_ptr[].last_ns_p = hit_point
                if not used_mnee:
                    _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, shadow_dir, dist * Float32(0.9999), contrib, guide_write)

# ── ReSTIR DI (Phase 2, restir_di.mojo's DIReservoir/di_target_pdf) ─────────
# Plain RIS: M candidates from the existing light sampler, weighted p̂/q,
# streamed into a reservoir (2.1+2.2); one shadow ray resolves the winner
# (2.6), MIS-weighted against BSDF sampling exactly like ordinary NEE (2.7).
# No temporal/spatial reuse yet (2.3/2.5 need persistent per-pixel GPU state
# across real interactive-viewer frames -- see restir_di.mojo's header and
# project_restir_migration.md memory for why that's scoped out for now).
# CPU-only (ctx.use_restir is only ever True from the CPU dispatch path,
# see ShadeContext's own docstring) -- traces the winner's shadow ray
# INLINE (_shadow_contribute[False]) rather than through Phase 0.4's
# deferred machinery the plan's 2.6 names: that machinery is GPU-only
# (shadow_buf lives in GpuSceneHandle), nothing CPU-side to defer into yet.
comptime DI_RIS_CANDIDATES: Int = 16

def _di_sample_candidate(
    ctx: ShadeContext, hit_point: Vec3f, mut pcg: PCG32,
) -> Tuple[Bool, Int32, Vec3f, Vec3f, RGB, Float32,
           Vec3f, Vec3f]:
    """One RIS candidate draw: (valid, light_idx, sample_point,
    light_normal, Le, generation_pdf, ldp_du, ldp_dv). Mirrors
    _sample_area_light_nee's math but keeps the sampled point/normal the
    reservoir payload needs -- that function only returns the derived
    wi/dist, not the point itself. The trailing tangent pair is the light
    surface's own parameterization at the sampled point, kept (rather than
    discarded as `_, _`) so a resampled winner can still run MNEE in
    di_resolve -- see DIReservoir's ldp_du/ldp_dv docs."""
    var zero3 = Vec3f(Float32(0))
    var invalid = (False, Int32(-1), zero3, zero3, RGB(Float32(0)), Float32(0), zero3, zero3)
    if ctx.lights.area_light_count == 0:
        return invalid
    var u_light = pcg.next_float()
    var ls_result = light_sampler_sample(ctx.lights.light_sampler, u_light)
    var light_idx = ls_result[0]
    var light_sel_pdf = ls_result[1]
    var al = ctx.lights.area_lights[light_idx]
    var r1 = pcg.next_float()
    var r2 = pcg.next_float()
    var (light_point, light_normal, ldp_du_v, ldp_dv_v) = _sample_light_point_and_normal(ctx, al, r1, r2, pcg)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    if dist_sq <= Float32(1e-8) or al.total_area <= Float32(0.0):
        return invalid
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_l = -dot(light_normal, wi)
    if cos_l <= Float32(0.0):
        return invalid
    var pdf = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf <= Float32(0.0):
        return invalid
    return (True, Int32(light_idx), light_point, light_normal, al.emission, pdf, ldp_du_v, ldp_dv_v)

def di_generate_reservoir(
    ctx: ShadeContext, hit_point: Vec3f, normal: Vec3f,
    alb: RGB, mut pcg: PCG32,
) -> DIReservoir:
    """Phase 2.1+2.2: stream DI_RIS_CANDIDATES area-light samples through
    weighted reservoir sampling, weight = p̂/q per candidate (reservoir.mojo).
    Visibility deliberately excluded (2.1) -- resolved once for the winner
    by di_resolve."""
    var res = di_reservoir_init()
    for _ in range(DI_RIS_CANDIDATES):
        var (valid, light_idx, sample_point, light_normal, le, gen_pdf, ldp_du_v, ldp_dv_v) = _di_sample_candidate(ctx, hit_point, pcg)
        if not valid:
            # A rejected candidate is still a REAL draw from q -- it just
            # landed where the target function is zero (light facing away,
            # cos_l <= 0, degenerate distance). RIS's W = w_sum / (M * p_hat)
            # needs M = the number of candidates DRAWN, not the number that
            # happened to be non-zero, so its zero weight must still be
            # streamed. Skipping this (an early `continue`) silently divides
            # by too small an M and inflates every W: on staircase2 roughly
            # 40% of candidates are backfacing, so m came out ~9 instead of
            # 16 and W was ~1.8x too large. reservoir_update with weight 0
            # returns before touching w_sum (and before reading `u`, hence
            # the dummy -- deliberately NOT a pcg draw, so the RNG stream is
            # identical to before this fix), but still does the m += 1 that
            # is the entire point here.
            _ = reservoir_update(res.state, Float32(0.0), Float32(0.0))
            continue
        var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
        var weight = p_hat / gen_pdf
        var accept = reservoir_update(res.state, weight, pcg.next_float())
        if accept:
            res.light_idx = light_idx
            res.sample_point = sample_point
            res.light_normal = light_normal
            res.ldp_du = ldp_du_v
            res.ldp_dv = ldp_dv_v
            res.le = le
    return res

def di_resolve(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut res: DIReservoir,
    z_norm: Float32 = Float32(-1.0),
):
    """Phase 2.6+2.7: finalize the reservoir's RIS weight, then trace ONE
    shadow ray for the winner and add its MIS-weighted contribution. The
    resampled winner is treated as a single light-technique sample for MIS
    purposes against BSDF sampling (same balance-heuristic pattern as
    ordinary NEE) -- standard RIS/ReSTIR practice; the RIS weight W already
    accounts for the M-candidate resampling itself.

    `z_norm` is forwarded verbatim to reservoir_finalize: negative (the
    default) means "divide by state.m", correct for single-domain
    combination; spatial reuse passes Bitterli 2020 Algorithm 6's Z
    instead. See di_temporal_step for how Z is accumulated."""
    if res.light_idx < Int32(0):
        return
    var p_hat = di_target_pdf(hit_point, normal, alb, res.sample_point, res.light_normal, res.le)
    reservoir_finalize(res.state, p_hat, z_norm)
    # Defensive weight clamp -- see DI_MAX_FINALIZED_WEIGHT's own comment
    # for the full mechanism (a real, live-traced bug: unbounded growth via
    # reservoir_combine's own feedback of a source's finalized state.w).
    # Must happen BEFORE the res.state.w <= 0 check below (clamping a
    # positive value can never make it <= 0) and BEFORE this frame's
    # result is stored for the next frame/neighbor to reuse.
    if res.state.w > DI_MAX_FINALIZED_WEIGHT:
        res.state.w = DI_MAX_FINALIZED_WEIGHT
    if res.state.w <= Float32(0.0):
        return
    var to_light = res.sample_point - hit_point
    var dist_sq = dot(to_light, to_light)
    if dist_sq <= Float32(1e-8):
        return
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    var cos_l = -dot(res.light_normal, wi)
    if cos_s <= Float32(0.0) or cos_l <= Float32(0.0):
        return
    var al = ctx.lights.area_lights[Int(res.light_idx)]
    if al.total_area <= Float32(0.0):
        return
    var light_sel_pdf = light_sampler_pdf(ctx.lights.light_sampler, res.light_idx)
    var pdf_light = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf_light <= Float32(0.0):
        return
    var pdf_bsdf = bxdf_pdf_diffuse(cos_s)
    var mis_w = power_heuristic(pdf_light, pdf_bsdf)
    # NOT cos_s*cos_l/dist_sq: pdf_light is a SOLID-ANGLE pdf (the
    # dist_sq/cos_l area-to-solid-angle Jacobian is already baked into its
    # own definition above). res.state.w (from reservoir_finalize) already
    # has a 1/p_hat(y) factor, and di_target_pdf's p_hat is built from the
    # FULL g=cos_s*cos_l/dist_sq -- multiplying by g again here double-counts
    # that Jacobian. Verified via the M=1 degenerate case (DI_RIS_CANDIDATES=1
    # forces exactly one candidate, always accepted): there, W collapses
    # algebraically to plain 1/pdf_light regardless of p_hat's shape (p_hat
    # cancels out of reservoir_finalize's w_sum/(M*p_hat) when w_sum IS
    # p_hat/gen_pdf), so contrib MUST reduce to _nee_area_lights's own
    # proven-correct cos_s/pdf_light formula -- which has no separate g
    # factor. Confirmed empirically too: adding g turned a ~3.5% mean
    # brightness bias into a ~28% one on staircase2 (400x400, vs a 128spp
    # plain-NEE reference).
    var contrib = path_ptr[].throughput * bxdf_eval_diffuse(alb) * res.le * (cos_s * res.state.w * mis_w)

    # MNEE, shared verbatim with _nee_area_lights. Without this the ReSTIR
    # path lost every bit of light reaching a surface THROUGH glass (the
    # straight shadow ray below is simply occluded by the dielectric), which
    # measured ~9% of staircase2's mean, spatially concentrated on the wall
    # and treads behind its glass railing.
    #
    # Measure conversion is the one thing that differs from the plain-NEE
    # caller: MNEE divides by an AREA-measure light pdf, but res.state.w
    # estimates 1/q in SOLID-ANGLE measure (_di_sample_candidate's gen_pdf
    # carries the dist^2/cos_l factor). Since q_sa = q_A * dist^2/cos_l,
    # 1/q_A = w * dist^2/cos_l, using the STRAIGHT-LINE dist/cos_l that
    # gen_pdf itself was built from (NOT the refracted path's geometry --
    # the manifold walk supplies its own G, and this factor is only undoing
    # the sampling measure, not describing the transport).
    var used_mnee = _mnee_area_light_contribute(
        path_ptr, ctx, normal, hit_point, alb, wi, dist,
        res.sample_point, res.ldp_du, res.ldp_dv, al,
        res.state.w * dist_sq / cos_l, Float32(1.0))
    if not used_mnee:
        _shadow_contribute[False](path_ptr, ctx, hit_point, wi, dist * Float32(0.9999), contrib)

# ── ReSTIR GI (Phase 4.1's remaining half: candidate generation) ────────────
# Scoped to x2 (the reconnection vertex) ALSO being diffuse -- mirrors how
# Phase 2 restricted x1 to diffuse first. This bounds the touch point to
# _shade_diffuse_nee alone: if bounce 1 lands on any other material, no GI
# candidate is generated for this path/sample at all (falls back to
# whatever that material's own shading does, completely unaffected). See
# project_restir_migration memory for why a fully general version (any
# material at x2) is real, separate, deferred work.
def _gi_generate_recon_candidate(
    ctx: ShadeContext, hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut pcg: PCG32,
) -> GIReservoir:
    """Generates x2's own contribution to Lo(x2): ONE area-light NEE sample
    (mirrors _di_sample_candidate's light pick, resolved as a plain single-
    sample estimate, NOT resampled via RIS -- ReSTIR GI's reconnection
    vertex gets ordinary NEE, not its own reservoir; the OUTER GI reservoir
    resamples over whole path suffixes, not over x2's own light picks).

    Explicit, documented scope limitation: Lo(x2) as computed here is
    area-light NEE ONLY -- no env/infinite/distant/point light contribution
    reaching x2 directly, and no further indirect bounces beyond x2. This
    is a real, honest simplification (matching Phase 2's own initial
    diffuse-and-area-lights-only scope), not a bug -- whoever generalizes
    this later should extend it the same incremental way Phase 2 itself
    was generalized, not treat this as a placeholder to silently rely on.

    No MIS weight against BSDF sampling is applied to the drawn light
    sample: since the complementary BSDF-sampling technique's own
    contribution is NOT separately included in this restricted definition
    of Lo, applying a MIS weight here would introduce a systematic
    negative bias rather than reduce variance -- correct only when treated
    as the sole technique for this quantity, which is exactly the case
    here.

    Returns an UNSTREAMED candidate (state left at reservoir_state_init())
    -- valid=0 when there are no area lights or the drawn sample is
    degenerate/backfacing at x2 (`normal`/`hit_point` here are x2's own).
    The caller must compute the real RIS weight using x1's OWN hit_point/
    normal/alb (not available here -- this runs at x2's shading call) via
    gi_target_pdf, then stream it with reservoir_update, before this
    becomes usable in gi_temporal_spatial_combine."""
    var (ok, light_idx, sample_point, light_normal, le, gen_pdf, ldp_du_v, ldp_dv_v) = _di_sample_candidate(ctx, hit_point, pcg)
    if not ok:
        return gi_reservoir_init()
    var to_light = sample_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    if cos_s <= Float32(0.0):
        return gi_reservoir_init()
    var lo = RGB(Float32(0.0))
    var shadow_ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(wi[0], wi[1], wi[2]))
    if not any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, shadow_ray, dist * Float32(0.9999),
                              ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                              ctx.lights.spheres, ctx.lights.sphere_count):
        lo = bxdf_eval_diffuse(alb) * le * (cos_s / gen_pdf)
    var res = gi_reservoir_init()
    res.recon_point = hit_point
    res.recon_normal = normal
    res.lo = lo
    res.recon_is_delta = Int8(0)  # x2 is diffuse by construction (only called from _shade_diffuse_nee)
    res.valid = Int8(1)
    return res

def gi_resolve(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    throughput: RGB,
    res: GIReservoir,
):
    """Phase 4's resolution step, mirrors di_resolve: trace ONE shadow ray
    between x1 (hit_point/normal/alb -- x1's OWN data, passed in explicitly
    since this runs at bounce 1's shading call where x1 is long out of
    scope, see GIPendingX1) and the reservoir's finalized winning
    reconnection vertex, adding its RIS-weighted contribution to
    path_ptr[].estimate if visible. Assumes gi_temporal_spatial_combine has
    ALREADY been called on `res` -- state.w is a finalized RIS correction
    weight (reservoir_finalize's output), not a raw stream sum.

    `throughput` MUST be the path's throughput as it EXISTED ENTERING x1
    (before x1's own BSDF sample was applied) -- NOT path_ptr[].throughput
    read fresh here, which by bounce 1 already has x1's ORIGINAL sampled
    continuation direction's f/cos/pdf baked in. That original direction is
    irrelevant to the RESAMPLED reconnection direction this function
    connects along; using it caused a real, shipped ~10x energy inflation,
    caught only once interactive mode's temporal reuse was render-tested
    for the first time (see GIPendingX1's own docstring for the full story
    and why the earlier unit-level tests couldn't have caught it). Callers
    must pass the GIPendingX1 snapshot's own `throughput` field.

    No MIS against BSDF sampling, unlike di_resolve: the complementary
    "reach x2 anyway by continuing the path" contribution is exactly what
    THIS reservoir sample already represents (there is no separate,
    parallel BSDF-sampling estimate of the same x1->x2 connection to weigh
    against here) -- same reasoning as _gi_generate_recon_candidate's own
    no-MIS choice, applied one level up.

    Formula derivation is identical to di_resolve's own (verified there via
    the M=1 degenerate case): contrib = throughput * f(x1) * res.lo *
    (cos_s * state.w), NOT cos_s*cos_x2/dist_sq*state.w -- gi_target_pdf's
    p_hat already bakes in the full G=cos_x1*cos_x2/dist_sq term, and
    state.w carries a 1/p_hat(winner) factor that would double-count G if
    multiplied in again here."""
    if res.valid == Int8(0) or res.state.w <= Float32(0.0):
        return
    var to_recon = res.recon_point - hit_point
    var dist_sq = dot(to_recon, to_recon)
    if dist_sq <= Float32(1e-8):
        return
    var dist = sqrt(dist_sq)
    var wi = to_recon * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    var cos_x2 = -dot(res.recon_normal, wi)
    if cos_s <= Float32(0.0) or cos_x2 <= Float32(0.0):
        return
    var contrib = throughput * bxdf_eval_diffuse(alb) * res.lo * (cos_s * res.state.w)
    _shadow_contribute[False](path_ptr, ctx, hit_point, wi, dist * Float32(0.9999), contrib)

# Temporal history clamp. The migration plan (and Bitterli et al. 2020)
# suggest ~20x the per-frame candidate count; measured here, 4x is better,
# so this deliberately departs from that guideline. staircase2 400x400,
# interactive 1spp/frame, MSE vs a 128spp plain-NEE reference:
#
#   temporal only, cap=20x (320):  0.001346 @128fr, converges 2.79x
#   temporal only, cap= 4x  (64):  0.001035 @128fr, converges 3.41x
#
# 1.30x better, with convergence rising to near plain NEE's own 3.75x.
# Mechanism: a reservoir pinned at a large cap gives each frame's fresh
# candidates only 16/320 ~ 5% of the total confidence, so it clings to
# recycled history and the effective independent-sample count grows far
# slower than m suggests. At 4x, fresh candidates carry 25%.
#
# Caveat worth respecting before trusting this number elsewhere: it is one
# scene, measured in the STATIC-camera accumulation regime, where trading
# sample independence for sample quality is a bad deal because the film is
# already averaging many frames. The cap's real job is bounding staleness
# across camera motion -- and render_interactive clears reservoirs outright
# on a camera move, so it rarely binds there at all. Re-measure before
# assuming 4x is right for a genuinely dynamic session.
comptime DI_TEMPORAL_M_CAP: Float32 = Float32(4 * DI_RIS_CANDIDATES)
# Phase 2.5 spatial reuse tuning, per the plan's own "k ~= 3-5 neighbors"
# and G-buffer rejection thresholds (docs/A2_restir_migration_plan.md).
#
# ENABLED. Getting here took three wrong verdicts, all caused by
# benchmarking in the wrong regime, so the reasoning is recorded in full.
#
# ReSTIR trades sample INDEPENDENCE for sample QUALITY. That pays when you
# display frame N; it costs when you AVERAGE N frames, because correlated
# frames do not average down. `--interactive-frames N` accumulates, so
# measuring at N=128 measures the regime ReSTIR is not for. Every earlier
# "ReSTIR loses" number here came from N=16/128.
#
# Measured on Scenes/restir-manylights.pbrt (256 equal-power lights, so
# the power-weighted CDF degenerates to uniform selection -- the case RIS
# exists to beat; ~500x spread in per-light contribution; all diffuse,
# maxdepth 1, so 100% of the image is inside ReSTIR's scope):
#
#   frames    plain    temporal   temporal+k=4    ReSTIR vs plain
#        1  0.07201     0.05772        0.05772    0.802x  WIN
#        2  0.02725     0.02152        0.02140    0.785x  WIN
#        4  0.01107     0.00934        0.00909    0.822x  WIN
#        8  0.00530     0.00513        0.00464    0.876x  WIN
#       16  0.00275     0.00316        0.00265    0.966x  WIN (spatial rescues it)
#
# Spatial contributes nothing at frame 1 (no neighbour has history yet)
# and grows to 0.838x of temporal-only by frame 16, turning what would be
# a 1.153x LOSS into a 0.966x win. It is off only in the sense that its
# benefit needs a frame or two to appear.
#
# Two earlier conclusions this overturned, both from bad benchmarks:
#   * "spatial is not worth enabling" -- measured on staircase2 (13
#     lights) and zero-day at 16/128 frames. Wrong on both axes.
#   * "ReSTIR never beats plain NEE" -- pure RIS with no reuse at all
#     (batch --restir) is 0.486x/0.499x on this scene, i.e. HALF the
#     error. The algorithm was always working; the benchmark was not.
#
# Also disproved here: the visibility theory (that RIS resamples toward
# unshadowed-looking-but-occluded lights). The generator emits an
# occluder-free variant, and ReSTIR behaves the same with and without
# occluders, so visibility is not what was limiting it.
comptime DI_SPATIAL_NEIGHBORS: Int = 4
# Fixed-size scratch for the Z pass, which must revisit each combined
# neighbour after the winner is known. +1 so the array is never zero-sized
# when someone sets DI_SPATIAL_NEIGHBORS = 0 to disable spatial reuse.
comptime DI_SPATIAL_SLOTS: Int = DI_SPATIAL_NEIGHBORS + 1
comptime DI_SPATIAL_RADIUS_PX: Float32 = Float32(20.0)
comptime DI_SPATIAL_NORMAL_DOT_MIN: Float32 = Float32(0.9)
comptime DI_SPATIAL_DEPTH_REL_MAX: Float32 = Float32(0.1)
# Defensive cap on a finalized reservoir's own state.w -- same fix, same
# root cause, as restir_gi.mojo's GI_MAX_FINALIZED_WEIGHT (see that
# constant's own comment for the full mechanism). reservoir_combine's
# weight formula (reservoir.mojo) multiplies in a SOURCE reservoir's own
# already-finalized state.w; without a bound, one anomalous value (root
# cause here: a stored light sample near-degenerate relative to a
# DIFFERENT pixel's own shading point -- e.g. a light sample point near
# the light mesh's own edge, viewed at grazing incidence from some pixel,
# not fully caught by the existing G-buffer rejection heuristics) gets
# fed back into every future combine that reuses it, compounding into
# unbounded growth. Confirmed via live tracing on cornell-box
# (--restir alone, no GI): w_sum grew ~1.3-1.7x per frame, reaching
# >600 billion within 30-40 frames while z_norm stayed roughly constant
# (~80) -- textbook exponential compounding, not a one-shot magnitude
# issue (this is why DI, unlike GI, WASN'T caught by this project's
# earlier extensive Phase 2 validation: that validation used interactive
# runs of a bounded frame count, per the plan's own noise-floor
# methodology, that happened to stay short enough to not yet manifest).
comptime DI_MAX_FINALIZED_WEIGHT: Float32 = Float32(10.0)

def di_temporal_step(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut pcg: PCG32,
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
):
    """Phase 2.1/2.2/2.3/2.5/2.6/2.7: generate this frame's fresh RIS
    candidates (di_generate_reservoir), temporally combine with the previous
    frame's stored reservoir at the SAME pixel via IDENTITY reprojection
    (fact #2, docs/A2_restir_migration_plan.md -- static camera means no
    shift/Jacobian is needed, target_pdf is just di_target_pdf re-evaluated
    at this frame's hit_point using the previous winner's stored candidate),
    then spatially combine with DI_SPATIAL_NEIGHBORS random neighbor pixels'
    PREVIOUS-frame reservoirs (restir_io.read again, not this frame's --
    render_all_tiles parallelizes per-tile, so a same-frame neighbor read
    would race against whichever thread hasn't shaded it yet), each gated by
    a G-buffer rejection test. The "reconnection shift" the plan names is
    exactly di_target_pdf re-evaluated at THIS pixel's hit_point/normal/alb
    using the neighbor's stored sample_point/light_normal/le -- same
    mechanism already used for temporal reuse above, no separate Jacobian
    code needed (di_target_pdf's cos_l/dist_sq terms ARE that Jacobian).
    Resolves one shadow ray for the final combined winner, then M-caps and
    stores to restir_io.write for the next frame to read. Falls back to
    plain single-frame RIS (no persistence, no spatial reuse) when
    restir_io isn't real -- the batch (non-interactive) --restir path, which
    has no cross-frame concept."""
    var res = di_generate_reservoir(ctx, hit_point, normal, alb, pcg)
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(restir_io.read)
    # Z-normalization bookkeeping, declared at function scope because the Z
    # pass below runs after the reuse block closes: which neighbour domains
    # took part, each one's confidence, and the confidence contributed by
    # domains identical to this pixel's own (see where it is assigned).
    var nb_px_seen = InlineArray[Int32, DI_SPATIAL_SLOTS](fill=Int32(-1))
    var nb_m_seen = InlineArray[Float32, DI_SPATIAL_SLOTS](fill=Float32(0))
    var nb_seen = 0
    var m_same_domain = Float32(0.0)
    if has_temporal:
        var prev = restir_io.read[pixel_idx]
        if prev.light_idx >= Int32(0):
            var p_hat_prev_here = di_target_pdf(hit_point, normal, alb, prev.sample_point, prev.light_normal, prev.le)
            var accept = reservoir_combine(res.state, prev.state, p_hat_prev_here, pcg.next_float())
            if accept:
                res.light_idx = prev.light_idx
                res.sample_point = prev.sample_point
                res.light_normal = prev.light_normal
                res.ldp_du = prev.ldp_du
                res.ldp_dv = prev.ldp_dv
                res.le = prev.le

        # Confidence accumulated so far comes from THIS pixel's own
        # candidates plus temporal reuse. Under identity reprojection the
        # previous frame's domain IS this pixel's domain (static camera), so
        # every bit of it can produce the winner and it all counts toward Z
        # unconditionally. Snapshot it before spatial combination starts,
        # since reservoir_combine grows state.m as it goes.
        m_same_domain = res.state.m

        if _is_real_ptr(restir_io.gbuf_normal) and restir_io.frame_w > Int32(0) and restir_io.frame_h > Int32(0):
            var self_px = Int32(pixel_idx) % restir_io.frame_w
            var self_py = Int32(pixel_idx) // restir_io.frame_w
            var self_depth = restir_io.gbuf_depth[pixel_idx]
            var self_mat = restir_io.gbuf_material_id[pixel_idx]
            for _ in range(DI_SPATIAL_NEIGHBORS):
                var ang = pcg.next_float() * Float32(6.283185307)
                var rad = sqrt(pcg.next_float()) * DI_SPATIAL_RADIUS_PX
                var nx = self_px + Int32(cos(ang) * rad)
                var ny = self_py + Int32(sin(ang) * rad)
                if nx < Int32(0) or nx >= restir_io.frame_w or ny < Int32(0) or ny >= restir_io.frame_h:
                    continue
                var n_idx = Int(ny * restir_io.frame_w + nx)
                if n_idx == pixel_idx:
                    continue
                var n_off = n_idx * 3
                var n_normal = Vec3f(
                    restir_io.gbuf_normal[n_off], restir_io.gbuf_normal[n_off + 1], restir_io.gbuf_normal[n_off + 2])
                if dot(n_normal, normal) < DI_SPATIAL_NORMAL_DOT_MIN:
                    continue
                var n_depth = restir_io.gbuf_depth[n_idx]
                if self_depth <= Float32(0.0) or abs(n_depth - self_depth) > DI_SPATIAL_DEPTH_REL_MAX * self_depth:
                    continue
                if restir_io.gbuf_material_id[n_idx] != self_mat:
                    continue
                var nb = restir_io.read[n_idx]
                if nb.light_idx < Int32(0):
                    continue
                var p_hat_nb_here = di_target_pdf(hit_point, normal, alb, nb.sample_point, nb.light_normal, nb.le)
                # Remember every neighbour actually folded in (winner or
                # not) -- Z is a property of which DOMAINS took part, not of
                # which one happened to win.
                if nb_seen < DI_SPATIAL_SLOTS:
                    nb_px_seen[nb_seen] = Int32(n_idx)
                    nb_m_seen[nb_seen] = nb.state.m
                    nb_seen += 1
                var accept_nb = reservoir_combine(res.state, nb.state, p_hat_nb_here, pcg.next_float())
                if accept_nb:
                    res.light_idx = nb.light_idx
                    res.sample_point = nb.sample_point
                    res.light_normal = nb.light_normal
                    res.ldp_du = nb.ldp_du
                    res.ldp_dv = nb.ldp_dv
                    res.le = nb.le

    # ── Z normalization (Bitterli et al. 2020, Algorithm 6) ───────────────
    # Dividing W by the full accumulated m assumes every combined reservoir
    # could have produced the chosen sample. Across DIFFERENT pixels that is
    # false: a neighbour whose surface faces away from the winning light
    # never could have generated it, yet its confidence still inflates the
    # denominator -- the systematic under-weighting that makes Algorithm 4
    # biased. Z instead counts only the domains that genuinely contain the
    # winner, which is the whole difference between the biased and unbiased
    # variants.
    #
    # "Domain i contains y" is tested by re-evaluating the target function
    # at neighbour i's OWN shading point (its world position + normal from
    # the Phase 0.3 G-buffer) rather than at this pixel's -- that is what
    # makes it a statement about neighbour i's integrand and not merely a
    # restatement of our own. Only the SIGN matters (>0 vs 0), so passing
    # this pixel's albedo is harmless: albedo scales p_hat, it cannot change
    # whether it is zero (and an all-black albedo contributes nothing under
    # either variant anyway). The test is therefore purely geometric --
    # exactly the cos_s>0 / cos_l>0 / dist>0 conditions inside di_target_pdf.
    var z_norm = Float32(-1.0)
    if nb_seen > 0 and _is_real_ptr(restir_io.gbuf_world_pos):
        var z = m_same_domain
        for i in range(nb_seen):
            var np_off = Int(nb_px_seen[i]) * 3
            var n_hit = Vec3f(
                restir_io.gbuf_world_pos[np_off],
                restir_io.gbuf_world_pos[np_off + 1],
                restir_io.gbuf_world_pos[np_off + 2])
            var n_nrm = Vec3f(
                restir_io.gbuf_normal[np_off],
                restir_io.gbuf_normal[np_off + 1],
                restir_io.gbuf_normal[np_off + 2])
            if di_target_pdf(n_hit, n_nrm, alb, res.sample_point, res.light_normal, res.le) > Float32(0.0):
                z += nb_m_seen[i]
        z_norm = z

    di_resolve(path_ptr, ctx, hit_point, normal, alb, res, z_norm)
    if has_temporal:
        # Note the ordering: state.m still holds the TRUE accumulated
        # confidence here (z_norm renormalized only W, never m), so the
        # M-cap and the stored history stay correct for the next frame.
        reservoir_cap_confidence(res.state, DI_TEMPORAL_M_CAP)
        restir_io.write[pixel_idx] = res

def _shade_diffuse_nee[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    wo: Vec3f,
    u_light: Float32,
    u_bary1: Float32,
    u_bary2: Float32,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    # NEE sampling asymmetry: area lights use CDF-weighted selection (one light per
    # bounce, weight = power), while infinite/env lights are ALL sampled every bounce.
    # Rationale: area lights are finite and numerous (N can be large), so stochastic
    # selection with MIS is necessary. Infinite lights are typically 1-2 env maps,
    # and their contribution is often dominant — sampling all of them every bounce
    # costs little and avoids the variance of single-sample env selection. At N>2
    # env lights this asymmetry would need revisiting.

    # ── Area light NEE ────────────────────────────────────────────────────────
    # Phase 2 (docs/A2_restir_migration_plan.md): ReSTIR DI's RIS reservoir
    # replaces plain one-sample NEE for area lights, primary bounce only
    # (bounce 0 -- matches every real ReSTIR DI implementation's scope; it's
    # a screen-space technique over the primary G-buffer, not a per-bounce
    # one). Deliberately does NOT get MNEE's glass-caustic probing that
    # _nee_area_lights below has -- restir_di.mojo's target function assumes
    # an unoccluded straight shadow ray, same simplifying assumption as
    # every other non-diffuse material's NEE in this file.
    #
    # ReSTIR SMS (Phase 6) is threaded into _nee_area_lights below, scoped
    # to bounce 0 ONLY (sms_io is only passed through when bounce==0) --
    # same screen-space-G-buffer reasoning as ReSTIR DI's own bounce-0
    # scope: sms_temporal_step's per-pixel reservoir reuse only makes sense
    # where hit_point is deterministic per pixel under a static camera,
    # true at the primary hit and NOT at deeper bounces (whose hit points
    # vary sample to sample even with a fixed camera). Known, documented
    # gap: when ctx.use_restir is ALSO active, di_temporal_step handles
    # bounce 0 instead of _nee_area_lights, so ReSTIR SMS's bounce-0 reuse
    # does not run in that combination yet -- same class of gap as MNEE's
    # own pre-existing "no glass probing under ReSTIR DI's bounce 0"
    # limitation noted above, not a new one. --sms-restir without --restir
    # is unaffected.
    var sms_io_this_bounce = sms_io if path_ptr[].bounce == Int32(0) else sms_reservoir_io_null()
    if ctx.use_restir and path_ptr[].bounce == Int32(0):
        di_temporal_step(path_ptr, ctx, hit_point, normal, alb, pcg, restir_io, pixel_idx)
    else:
        _nee_area_lights[enqueue_shadow](path_ptr, ctx, normal, hit_point, alb, u_light, u_bary1, u_bary2, pcg, guide_write,
            sms_io=sms_io_this_bounce, pixel_idx=pixel_idx)

    # ── ReSTIR GI (Phase 4.1's remaining half) ───────────────────────────────
    # x1 is diffuse (this function only runs for diffuse hits) and bounce 0:
    # mark this path as awaiting a possible reconnection vertex at bounce 1,
    # carrying x1's own shading data forward (gi_target_pdf needs it later,
    # and it's out of scope by the time bounce 1's shade call runs). Real
    # only when ctx.gi_pending is a live buffer -- dangling on every call
    # site except shade_core_cpu_nee's, so this is a no-op everywhere else,
    # including every GPU kernel and every non-restir CPU render.
    if ctx.use_restir and path_ptr[].bounce == Int32(0) and _is_real_ptr(ctx.gi_pending):
        ctx.gi_pending[ctx.path_idx] = GIPendingX1(active=Int8(1), hit_point=hit_point, normal=normal, alb=alb, throughput=path_ptr[].throughput)
    elif (ctx.use_restir and path_ptr[].bounce == Int32(1) and _is_real_ptr(ctx.gi_pending)
          and ctx.gi_pending[ctx.path_idx].active == Int8(1)):
        # x2 is ALSO diffuse (same reasoning: this function only runs for
        # diffuse hits) -- exactly the scope this increment supports. Any
        # other bounce-1 material silently never reaches here at all (that
        # material's own shade_* function has no gi_pending-consuming logic),
        # so gi_pending[tid].active is simply left at 1 and never read again
        # -- harmless, see GIPendingX1's own docstring.
        var snap = ctx.gi_pending[ctx.path_idx]
        ctx.gi_pending[ctx.path_idx].active = Int8(0)
        var raw = _gi_generate_recon_candidate(ctx, hit_point, normal, alb, pcg)
        if raw.valid != Int8(0):
            var w = gi_target_pdf(snap.hit_point, snap.normal, snap.alb, raw.recon_point, raw.recon_normal, raw.lo)
            _ = reservoir_update(raw.state, w, pcg.next_float())
        # Combine with history/neighbors (Phase 4.2 -- a no-op fallback to a
        # plain single-candidate finalize when ctx.gi_io/pixel_idx aren't
        # real, matching di_temporal_step's own null-safety contract) then
        # resolve: one shadow ray between x1 (snap's own data) and the
        # combined winner, injected into path_ptr[].estimate if visible.
        gi_temporal_spatial_combine(raw, snap.hit_point, snap.normal, snap.alb, pcg, ctx.gi_io, pixel_idx)
        gi_resolve(path_ptr, ctx, snap.hit_point, snap.normal, snap.alb, snap.throughput, raw)

    # ── Distant/point/sphere light NEE, via the shared Light interface
    # (bvh.mojo's LightSample samplers) + BxDF interface (bxdf.mojo's
    # _nee_weight_simple) — replacing 3 formerly hand-inlined blocks also
    # duplicated in _shade_conductor_nee/shade_hair. Infinite lights keep
    # their own dedicated _nee_infinite_light call below: its no-CDF
    # fallback deliberately uses cosine-weighted (not uniform-sphere)
    # sampling, a diffuse-specific optimization that must NOT be replaced
    # by the material-agnostic sampler. Area lights stay on _nee_area_lights
    # above (MNEE glass-refraction probing, also diffuse-specific).
    _nee_loop_simple[enqueue_shadow](path_ptr, ctx, normal, hit_point, alb, Float32(0.0), Int32(0), wo, pcg, guide_write)

    # ── Infinite (env-map) light NEE ──────────────────────────────────────────
    for inf_i in range(ctx.lights.infinite_count):
        _nee_infinite_light[enqueue_shadow](path_ptr, ctx, ctx.lights.infinite_lights[inf_i], normal, hit_point, alb, u_env1, u_env2, pcg, guide_write)


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
def shade_diffuse[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    var (gc, ok) = _build_geom_context_full[use_gpu](path_ptr, inter, mat, ctx)
    if not ok:
        path_ptr[].active = 0
        return
    var normal = gc.normal
    var hit_point = gc.hit_point
    var alb = gc.alb

    # pbrt's diffuse BRDF is zero when the viewer and the lit direction are in
    # opposite hemispheres of the SHADING normal (SameHemisphere). A normal map
    # (or, on low-poly meshes, plain vertex-normal interpolation) can tilt the
    # shading normal past the viewer at grazing angles. Falling back to the
    # geometric normal (always front-facing to wo, see GeomContext.geo_normal)
    # keeps these points lit instead of going black — this matters for any
    # textured infinite light (env map), not just the uniform-color case a
    # closed-form ambient add could shortcut.
    if dot(normal, gc.wo) <= Float32(0.0):
        normal = gc.geo_normal

    # Sampler design: diffuse uses Sobol for key visible decisions (light selection,
    # barycentrics, env direction, scatter, RR) and PCG for auxiliary draws that
    # don't benefit from low-discrepancy sequences (area-light triangle index,
    # sphere-light direction, infinite-light fallback, indirect-bounce scatter).
    # Specular materials (conductor, dielectric, coated_conductor, thin_dielectric)
    # use PCG only — their scatter decisions are near-deterministic at low roughness,
    # and they don't do NEE, so stratification yields negligible variance reduction.
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Pre-draw 8 Z-Sobol samples for this bounce's key decisions.
    # Dims are consecutive starting at path_ptr[].sampler_dim (which begins at 2).
    # Per-dimension scrambling is derived from pcgInc (unique per path).
    var ss = _draw_sobol_8(path_ptr, ctx.sobol_matrices)
    var u_light = ss.light
    var u_bary1 = ss.bary1
    var u_bary2 = ss.bary2
    var u_env1  = ss.env1
    var u_env2  = ss.env2
    var u_scat1 = ss.scat1
    var u_scat2 = ss.scat2
    var u_rr    = ss.rr

    _shade_diffuse_nee[use_gpu, enqueue_shadow](path_ptr, ctx, normal, hit_point, alb, gc.wo,
        u_light, u_bary1, u_bary2, u_env1, u_env2, pcg, guide_write, restir_io, pixel_idx, sms_io)

    # ── Scatter direction: 50/50 mixture of guide and cosine-weighted BSDF ──────
    # The guide is active when its energy pointer is a real allocation (Int > 1).
    # MIS balance heuristic: weight = f·cosθ / pdf_mix, where
    #   pdf_mix = 0.5·pdf_guide + 0.5·pdf_bsdf  and  f = alb/π (Lambertian).
    var dir: Vec3f
    var cos_theta: Float32
    var pdf_mix: Float32

    if guide_is_active(ctx.guide):
        var cell = guide_pos_to_cell(ctx.guide, Point3f(hit_point[0], hit_point[1], hit_point[2]))
        var bsdf_s = sample_cosine_hemisphere_world(u_scat1, u_scat2, normal)
        var bsdf_dir = bsdf_s[0]
        var pdf_b = bsdf_s[1]  # cos(theta)/π at bsdf_dir
        # Only apply 50/50 mixture when this cell has training data.
        # Empty cells return guide_pdf = 1/4π which is less than typical bsdf pdf,
        # inflating pdf_mix above pdf_b and increasing variance for no benefit.
        var has_guide = guide_cell_has_data(ctx.guide, cell)
        # α=0.25: 25% guide samples, 75% BSDF. MIS weights: α·pg + (1-α)·pb.
        # Low α limits the variance increase when the guide distribution is wrong.
        comptime GUIDE_ALPHA = Float32(0.25)
        comptime GUIDE_BETA  = Float32(0.75)
        if has_guide and pcg.next_float() < GUIDE_ALPHA:
            # Guide sample path
            var (gdx, gdy, gdz, pdf_g, guide_ok) = guide_sample(ctx.guide, cell, pcg.next_float(), pcg.next_float())
            var cos_g = gdx*normal[0] + gdy*normal[1] + gdz*normal[2]
            if guide_ok and cos_g > Float32(0.001):
                dir = Vec3f(gdx, gdy, gdz)
                cos_theta = cos_g
                pdf_mix = GUIDE_ALPHA*pdf_g + GUIDE_BETA*(cos_g / PI)
            else:
                # Guide direction below surface — fall back to BSDF with MIS
                dir = bsdf_dir
                cos_theta = pdf_b * PI
                var pg = guide_pdf(ctx.guide, cell, bsdf_dir[0], bsdf_dir[1], bsdf_dir[2])
                pdf_mix = GUIDE_ALPHA*pg + GUIDE_BETA*pdf_b
        elif has_guide:
            # BSDF sample with guide MIS correction
            dir = bsdf_dir
            cos_theta = pdf_b * PI
            var pg = guide_pdf(ctx.guide, cell, bsdf_dir[0], bsdf_dir[1], bsdf_dir[2])
            pdf_mix = GUIDE_ALPHA*pg + GUIDE_BETA*pdf_b
        else:
            # Cell outside AABB or empty — pure BSDF
            dir = bsdf_dir
            cos_theta = pdf_b * PI
            pdf_mix = pdf_b
    else:
        var bsdf_s = sample_cosine_hemisphere_world(u_scat1, u_scat2, normal)
        dir = bsdf_s[0]
        cos_theta = bsdf_s[1] * PI  # cos_theta = pdf * π
        pdf_mix = bsdf_s[1]

    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(dir[0], dir[1], dir[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = alb
    # Store mixture PDF for next-bounce MIS (area light hit, env light miss)
    path_ptr[].lastBsdfPdf = pdf_mix
    path_ptr[].specularBounce = Int8(0)
    # Weight = f·cosθ / pdf_mix = (alb/π)·cosθ / pdf_mix.
    # MIS balance heuristic is unbiased — no cap needed here.
    var _w = cos_theta / (PI * pdf_mix)
    path_ptr[].throughput *= alb * _w
    path_ptr[].bounce += 1

    _apply_russian_roulette(path_ptr, pcg, u_rr)


@always_inline
def shade_interface(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
):
    """Interface (null/passthrough) material: advance the ray through the surface.
    No scattering, no throughput change. Medium transition is handled externally:
    on CPU by rendering.mojo's medium-interface loop; on GPU by shade_interface_gpu."""
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + ray_dir * Float32(0.0002)
    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), path_ptr[].ray.direction)
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)


@always_inline
def _shade_dispatch[use_gpu: Bool, enqueue_shadow: Bool](
    mat: Material_C,
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    if mat.type == MatKind.diffuse:
        shade_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat, guide_write, restir_io, pixel_idx, sms_io)
    # Delta BSDFs (dielectric variants) need only triangle geometry — no NEE,
    # textures, or Sobol. Passing ctx.meshes directly keeps GPU kernel
    # argument counts minimal. Conductor/coated_conductor CAN be rough (not
    # delta), so they now take full ctx for NEE — see shade_conductor's
    # own docstring/_shade_conductor_nee.
    elif mat.type == MatKind.conductor:
        shade_conductor[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.dielectric:
        shade_dielectric[False](path_ptr, inter, ctx.meshes, mat, ctx.lights.spheres, ctx.tex_filenames, ctx.textures, ctx.n_textures)
    elif mat.type == MatKind.coated_diffuse:
        shade_coated_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.diffuse_transmit:
        shade_diffuse_transmission[use_gpu, enqueue_shadow](path_ptr, inter, ctx)
    elif mat.type == MatKind.coated_conductor:
        shade_coated_conductor[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.mix:
        shade_mix[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.thin_dielectric:
        shade_thin_dielectric(path_ptr, inter, ctx.meshes, mat, ctx.lights.spheres)
    elif mat.type == MatKind.interface:
        shade_interface(path_ptr, inter)
    elif mat.type == MatKind.hair:
        comptime if not use_gpu:
            shade_hair[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.measured:
        shade_measured[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    else:
        path_ptr[].active = 0


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
def shade_nee_core[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    # ── Miss handler: ray escaped — add infinite light and deactivate ──────────
    if inter.hit == 0:
        # Volume scatter sets hit=0 to skip surface shading but is not a true miss.
        if path_ptr[].volume_scattered == Int8(1):
            path_ptr[].volume_scattered = Int8(0)
            return
        var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        var miss_albedo = RGB(Float32(0.0))
        for inf_i in range(ctx.lights.infinite_count):
            var ilight = ctx.lights.infinite_lights[inf_i]
            # Transform world-space ray direction into light's local frame
            var w2l = ilight.world_to_light
            var ld_x = w2l[0]*ray_dir[0] + w2l[4]*ray_dir[1] + w2l[8]*ray_dir[2]
            var ld_y = w2l[1]*ray_dir[0] + w2l[5]*ray_dir[1] + w2l[9]*ray_dir[2]
            var ld_z = w2l[2]*ray_dir[0] + w2l[6]*ray_dir[1] + w2l[10]*ray_dir[2]
            var local_dir = Vec3f(ld_x, ld_y, ld_z)
            var env_rgb: RGB
            if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and ilight.cdf_w > Int32(0):
                # Bilinear lookup in GPU/CPU-resident pixels (cdf_w × cdf_h, 3 floats/pixel)
                var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                var ea_uv = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                var u = ea_uv[0]; var v = ea_uv[1]
                var fx = u * Float32(iw) - Float32(0.5)
                var fy = v * Float32(ih) - Float32(0.5)
                var x0 = Int(max(Float32(0), min(Float32(iw - 1), floor(fx))))
                var y0 = Int(max(Float32(0), min(Float32(ih - 1), floor(fy))))
                var x1 = min(x0 + 1, iw - 1)
                var y1 = min(y0 + 1, ih - 1)
                var wx = fx - Float32(x0); var wy = fy - Float32(y0)
                var r00 = ilight.pixels_ptr[(y0*iw+x0)*3+0]; var g00 = ilight.pixels_ptr[(y0*iw+x0)*3+1]; var b00 = ilight.pixels_ptr[(y0*iw+x0)*3+2]
                var r10 = ilight.pixels_ptr[(y0*iw+x1)*3+0]; var g10 = ilight.pixels_ptr[(y0*iw+x1)*3+1]; var b10 = ilight.pixels_ptr[(y0*iw+x1)*3+2]
                var r01 = ilight.pixels_ptr[(y1*iw+x0)*3+0]; var g01 = ilight.pixels_ptr[(y1*iw+x0)*3+1]; var b01 = ilight.pixels_ptr[(y1*iw+x0)*3+2]
                var r11 = ilight.pixels_ptr[(y1*iw+x1)*3+0]; var g11 = ilight.pixels_ptr[(y1*iw+x1)*3+1]; var b11 = ilight.pixels_ptr[(y1*iw+x1)*3+2]
                var tr = (Float32(1)-wx)*(Float32(1)-wy)*r00 + wx*(Float32(1)-wy)*r10 + (Float32(1)-wx)*wy*r01 + wx*wy*r11
                var tg = (Float32(1)-wx)*(Float32(1)-wy)*g00 + wx*(Float32(1)-wy)*g10 + (Float32(1)-wx)*wy*g01 + wx*wy*g11
                var tb = (Float32(1)-wx)*(Float32(1)-wy)*b00 + wx*(Float32(1)-wy)*b10 + (Float32(1)-wx)*wy*b01 + wx*wy*b11
                env_rgb = RGB(tr, tg, tb) * ilight.scale
            else:
                comptime if not use_gpu:
                    if ilight.tex_idx >= Int32(0):
                        var fname = ctx.tex_filenames[Int(ilight.tex_idx)]
                        var ea_uv2 = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                        var u = ea_uv2[0]; var v = ea_uv2[1]
                        var tr = alloc[Float32](3)
                        tr[0] = Float32(0.0); tr[1] = Float32(0.0); tr[2] = Float32(0.0)
                        _ = external_call["texture", Bool,
                            UnsafePointer[UInt8, MutExternalOrigin], Float32, Float32,
                            UnsafePointer[Float32, MutExternalOrigin]](fname, u, v, tr)
                        env_rgb = RGB(tr[0], tr[1], tr[2]) * ilight.scale
                        tr.free()
                    else:
                        env_rgb = ilight.scale
                else:
                    env_rgb = ilight.scale
            miss_albedo += env_rgb
            var mis_weight = Float32(1.0)
            # env_rgb_contrib defaults to the bilinear-filtered env_rgb (smooth
            # sky appearance for direct/specular escapes, where mis_weight stays
            # 1.0 and no pdf is involved). For MIS-weighted indirect escapes
            # (bounce>0, non-specular) with a CDF-textured light, it's
            # overridden below to the SAME nearest-texel value the pdf is
            # computed from — bilinear-blending the radiance while reading the
            # pdf from one sharp texel is a radiance/pdf registration mismatch
            # that biases (and adds fireflies to) this MIS term specifically.
            var env_rgb_contrib = env_rgb
            if path_ptr[].specularBounce == Int8(0) and path_ptr[].bounce > 0:
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                # Uniform env: NEE cosine-hemisphere samples it, so the light pdf
                # for this (cosine-sampled) direction equals pdf_bsdf -> MIS 0.5.
                # (Was INV_FOUR_PI, inconsistent with the NEE sampler.)
                var pdf_light = pdf_bsdf
                # Use CDF-based pdf when available (env-map importance sampling)
                if ilight.cdf_w > Int32(0) and ilight.cdf_h > Int32(0):
                        var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
                        var ea_uv3 = _equal_area_sphere_to_square(local_dir[0], local_dir[1], local_dir[2])
                        var u = ea_uv3[0]; var v = ea_uv3[1]
                        var px = Int(min(Float32(iw - 1), max(Float32(0.0), u * Float32(iw))))
                        var py = Int(min(Float32(ih - 1), max(Float32(0.0), v * Float32(ih))))
                        var marginal_base = ih + 1
                        var row_cdf_base = marginal_base + py * (iw + 1)
                        # pdf of this texel in the CDF (equal-area: uniform solid angle)
                        var dp_row = ilight.cdf_ptr[py + 1] - ilight.cdf_ptr[py]
                        var dp_col_base = row_cdf_base + px
                        var dp_col = ilight.cdf_ptr[dp_col_base + 1] - ilight.cdf_ptr[dp_col_base]
                        if dp_row > Float32(0.0):
                            pdf_light = dp_row * dp_col * Float32(iw) * Float32(ih) * INV_FOUR_PI
                        if _is_real_ptr(ilight.pixels_ptr):
                            var nr = ilight.pixels_ptr[(py*iw+px)*3+0]
                            var ng = ilight.pixels_ptr[(py*iw+px)*3+1]
                            var nb = ilight.pixels_ptr[(py*iw+px)*3+2]
                            env_rgb_contrib = RGB(nr, ng, nb) * ilight.scale
                mis_weight = power_heuristic(pdf_bsdf, pdf_light)
            path_ptr[].estimate += path_ptr[].throughput * env_rgb_contrib * mis_weight
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].albedo = miss_albedo
        path_ptr[].active = 0
        return

    var mat = ctx.materials[Int(inter.primId.materialIndex)]

    # ── Analytical sphere hit: collect emission and terminate, but ONLY for
    # actual area-light spheres. A non-emissive sphere (e.g. a medium-bounding
    # volume with Material "interface", or a regular conductor/diffuse
    # sphere) falls through to the normal material dispatch below, same as
    # any other primitive type — it used to unconditionally terminate here
    # regardless of isAreaLight, silently killing every ray that hit any
    # non-light sphere (e.g. smoke-plume's medium-bounding sphere never got
    # to run its "interface" material's pass-through logic).
    if inter.primId.type == Int8(4) and ctx.lights.sphere_count > 0:
        var sph_idx = Int(inter.primId.id1)
        var sph = ctx.lights.spheres[sph_idx]
        if sph.isAreaLight == Int8(1):
            if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
                path_ptr[].estimate += path_ptr[].throughput * sph.emission
            else:
                # MIS: BSDF pdf vs sphere solid-angle pdf from previous shading point
                var pdf_bsdf = path_ptr[].lastBsdfPdf
                var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
                var to_c_x = sph.center.x - ray_org[0]
                var to_c_y = sph.center.y - ray_org[1]
                var to_c_z = sph.center.z - ray_org[2]
                var dc_sq = to_c_x*to_c_x + to_c_y*to_c_y + to_c_z*to_c_z
                var sin2_max = sph.radius * sph.radius / dc_sq
                if sin2_max < Float32(1.0) and pdf_bsdf > Float32(0.0):
                    var cos_max = sqrt(Float32(1.0) - sin2_max)
                    var solid_angle = TWO_PI * (Float32(1.0) - cos_max)
                    var pdf_light = Float32(1.0) / (solid_angle * Float32(max(ctx.lights.sphere_count, 1)))
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += path_ptr[].throughput * sph.emission * w
            path_ptr[].active = 0
            return

    if inter.primId.type == Int8(3):
        # Area light triangle hit — use emission from AreaLight_C directly so
        # NamedMaterial area lights (mat.type == 1) also emit correctly.
        var al_idx = Int(inter.primId.id1)
        var al = ctx.lights.area_lights[al_idx]
        var emission = al.emission
        if path_ptr[].specularBounce == Int8(1) and path_ptr[].sms_covered == Int8(1):
            # This arrived through a specular chain from a vertex that
            # delegates such paths to MNEE/SMS. MNEE covers the segment to a
            # given emitter point exactly when a dielectric intervenes on it,
            # so ask that same question here, for the point actually hit: if
            # glass is in the way, MNEE already sampled this path family and
            # counting it again double-counts (see PathState_C.sms_covered).
            # If not, MNEE never had it and BSDF sampling is the only
            # strategy that does -- keep it.
            var hp = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) \
                     + Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z) * inter.tHit
            var seg = hp - path_ptr[].last_ns_p
            var seg_len = sqrt(dot(seg, seg))
            if seg_len > Float32(1e-6):
                var seg_dir = seg * (Float32(1.0) / seg_len)
                var pr_org = path_ptr[].last_ns_p + seg_dir * Float32(0.0001)
                var pr_ray = Ray_C(Point3f(pr_org[0], pr_org[1], pr_org[2]),
                                   Vec3f(seg_dir[0], seg_dir[1], seg_dir[2]))
                var pr_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
                var pr_store = InlineArray[Intersection_C, 1](fill=Intersection_C(
                    pr_prim, Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
                traverse_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, pr_ray,
                                   seg_len * Float32(0.999), pr_store.unsafe_ptr(),
                                   ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                                   ctx.lights.spheres, ctx.lights.sphere_count)
                var pr = pr_store[0]
                if pr.hit != Int8(0):
                    var pr_mat = ctx.materials[Int(pr.primId.materialIndex)]
                    if pr_mat.type == MatKind.dielectric or pr_mat.type == MatKind.thin_dielectric:
                        path_ptr[].active = 0
                        return
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].estimate += path_ptr[].throughput * emission
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(inter, ctx.meshes)
                var lp0 = Vec3f(lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
                var lp1 = Vec3f(lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
                var lp2 = Vec3f(lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
                var lnorm = cross(lp1 - lp0, lp2 - lp0)
                var lnlen = dot(lnorm, lnorm)
                if lnlen > Float32(0.0):
                    lnorm = lnorm * (Float32(1.0) / sqrt(lnlen))
                var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                var cos_l = -dot(lnorm, ray_dir)
                var dist  = inter.tHit
                var dist2 = dist * dist
                if cos_l > Float32(0.0) and al.total_area > Float32(0.0):
                    var ls = ctx.lights.light_sampler
                    var al_sel_pdf = ls.cdf[al_idx + 1] - ls.cdf[al_idx]
                    var pdf_light = dist2 * max(al_sel_pdf, Float32(1e-6)) / (cos_l * al.total_area)
                    var w = power_heuristic(pdf_bsdf, pdf_light)
                    path_ptr[].estimate += path_ptr[].throughput * emission * w
        path_ptr[].active = 0
        return

    if inter.primId.type == Int8(5) and mat.type == MatKind.area_light:
        # Emissive curve hit directly by the camera/bounce ray. Curves are
        # now explicitly NEE-sampled too (AreaLight_C.kind==1, see al_list
        # in finalize_scene) — symmetric with the type==3 triangle case
        # above, this MIS-weights against the curve's own selection pdf
        # instead of always crediting full emission (which would double-
        # count against the NEE strategy now that one exists).
        var emission = mat.emission
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].estimate += path_ptr[].throughput * emission
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var curve_idx = Int(inter.primId.id1)
                # Curve lights are rare (a handful of emissive strands at
                # most), so a linear scan for this curve's al_list slot is
                # cheap — no reverse index is threaded through PrimId_C.
                var al_idx = -1
                for li in range(ctx.lights.area_light_count):
                    var cand = ctx.lights.area_lights[li]
                    if cand.kind == Int8(1) and Int(cand.meshIdx) == curve_idx:
                        al_idx = li
                        break
                if al_idx >= 0:
                    var al = ctx.lights.area_lights[al_idx]
                    if al.total_area > Float32(0.0):
                        # Reconstruct the outward radial normal at the actual
                        # hit point from (u, v) — same formula shade_hair uses.
                        var curve = ctx.curves[curve_idx]
                        var h = max(Float32(-0.99), min(Float32(0.99), inter.u))
                        var v_global = inter.v
                        var piece = min(Int(curve.n_pieces) - 1, max(0, Int(v_global * Float32(curve.n_pieces))))
                        var (q0, q1, _, _) = curve_piece_endpoints(curve, piece)
                        var seg_axis = q1 - q0
                        var seg_len = sqrt(dot(seg_axis, seg_axis))
                        var tangent = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
                        if seg_len > Float32(1e-8):
                            tangent = seg_axis * (Float32(1.0) / seg_len)
                        var n_perp = _curve_perp_axis(tangent)
                        var b_perp0 = cross(tangent, n_perp)
                        var geo_normal = n_perp * h + b_perp0 * sqrt(max(Float32(0.0), Float32(1.0) - h*h))
                        var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
                        var cos_l = -dot(geo_normal, ray_dir)
                        var dist2 = inter.tHit * inter.tHit
                        if cos_l > Float32(0.0):
                            var ls = ctx.lights.light_sampler
                            var al_sel_pdf = ls.cdf[al_idx + 1] - ls.cdf[al_idx]
                            var pdf_light = dist2 * max(al_sel_pdf, Float32(1e-6)) / (cos_l * al.total_area)
                            var w = power_heuristic(pdf_bsdf, pdf_light)
                            path_ptr[].estimate += path_ptr[].throughput * emission * w
        path_ptr[].active = 0
        return

    comptime if use_gpu:
        # GPU: mark material for its dedicated per-material kernel
        path_ptr[].pending_mat = mat.type
    else:
        _shade_dispatch[False, enqueue_shadow](mat, path_ptr, inter, ctx, guide_write, restir_io, pixel_idx, sms_io)


@always_inline
def shade_core_cpu_nee(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    tid: Int,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    distantLightCount: Int,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    pointLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    infiniteLightCount: Int,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    sphereCount: Int,
    light_sampler: LightSampler_C,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    guide: GuideGrid,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
    guide_write: GuideGrid = null_guide(),
    spectral: SpectralHandle = null_spectral_handle(),
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutExternalOrigin] = UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
    use_restir: Bool = False,
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    gi_pending: UnsafePointer[GIPendingX1, MutExternalOrigin] = UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(),
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    nmaps: UnsafePointer[NormalSlopeMap_C, MutExternalOrigin] = UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=tex_filenames,
        textures=UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling(), n_textures=0,
        nmaps=nmaps,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutExternalOrigin].unsafe_dangling(),
        px_scale=Float32(0.0), sobol_matrices=sobol_matrices, guide=guide, use_restir=use_restir,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=spectral, measured_brdfs=measured_brdfs,
        gi_pending=gi_pending, gi_io=gi_io,
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=distantLightCount,
            point_lights=pointLights, point_count=pointLightCount,
            infinite_lights=infiniteLights, infinite_count=infiniteLightCount,
            spheres=spheres, sphere_count=sphereCount, light_sampler=light_sampler))
    shade_nee_core[False, False](path_ptr, inter, ctx, guide_write, restir_io, pixel_idx, sms_io)
