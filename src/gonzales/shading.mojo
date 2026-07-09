from std.math import sqrt, cos, sin, floor, acos, atan2, log2, exp, log, abs
from std.ffi import external_call
from std.memory import alloc
from .geometry import RGB, SampledSpectrum, Point3f, Point2f, Vec3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, MatKind, AreaLight_C, Sphere_C, Curve_C, CURVE_N_PIECES, curve_piece_endpoints, _curve_perp_axis, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, ShadowTask_C, LightSampler_C, light_sampler_sample, Instance_C, dot, cross, Frame, safe_sqrt, reflect, refract, schlick_fresnel, fr_dielectric, PI, TWO_PI, INV_PI, INV_FOUR_PI
from .bxdf import BxDFSample, GeomContext, SobolSamples8, BxDFFlags, bxdf_is_delta, bxdf_sample_conductor, bxdf_sample_coated_conductor, bxdf_sample_dielectric, bxdf_sample_thin_dielectric, bxdf_eval_diffuse, bxdf_pdf_diffuse, bxdf_sample_diffuse, bxdf_sample_diffuse_transmit, ggx_D, ggx_G1, ggx_G2, ggx_vndf_pdf, bxdf_eval_conductor_ggx, bxdf_pdf_conductor_ggx, _nee_weight_simple, _nee_weight_hair, _nee_weight_simple_via_spectral, _nee_weight_coated_coat_lobe, _nee_weight_coated_diffuse_base
from .rng import PCG32
from .bvh import BVH2Node, SceneDescriptor2_C, any_hit_bvh2_core, ray_sphere_hit, traverse_bvh2_core, HairLobeConstants, _hair_precompute, _hair_eval_lobes, _hair_sample_dir, curve_offset_eps, LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee, _sample_infinite_light_textured, _equal_area_square_to_sphere, _equal_area_sphere_to_square
from .sampling import power_heuristic, sample_cosine_hemisphere, sample_cosine_hemisphere_world, sample_ggx_vndf, sobol_sample, mix_bits_u64
from .transform import transform_normal_by_instance
from .guide import GuideGrid, guide_pos_to_cell, guide_pdf, guide_sample, guide_cell_has_data, guide_record, null_guide
from .spectrum import SpectralHandle, null_spectral_handle

@fieldwise_init
struct LightContext(Copyable, Movable):
    """Light source arrays + counts + sampler, used by NEE. Split out of
    ShadeContext (step 7) so the grouping of "which lights exist in the scene"
    is self-contained instead of interleaved with scene/texture/sampling
    fields — construction call sites build this with keyword args precisely
    because several fields share the same Int/pointer type and a positional
    transposition wouldn't be caught by the type checker."""
    var area_lights:      UnsafePointer[AreaLight_C, MutAnyOrigin]
    var area_light_count: Int
    var distant_lights:   UnsafePointer[DistantLight_C, MutAnyOrigin]
    var distant_count:    Int
    var point_lights:     UnsafePointer[PointLight_C, MutAnyOrigin]
    var point_count:      Int
    var infinite_lights:  UnsafePointer[InfiniteLight_C, MutAnyOrigin]
    var infinite_count:   Int
    var spheres:          UnsafePointer[Sphere_C, MutAnyOrigin]
    var sphere_count:     Int
    var light_sampler:    LightSampler_C

@fieldwise_init
struct ShadeContext:
    """Bundles scene/texture/sampling data pointers passed to shade_nee_core
    and material shaders. Light-source data lives in the nested `lights`
    LightContext (step 7)."""
    var path_idx:         Int
    var bvh2Nodes:        UnsafePointer[BVH2Node, MutAnyOrigin]
    var primIds:          UnsafePointer[PrimId_C, MutAnyOrigin]
    var meshes:           UnsafePointer[TriangleMesh_C, MutAnyOrigin]
    var curves:           UnsafePointer[Curve_C, MutAnyOrigin]
    var materials:        UnsafePointer[Material_C, MutAnyOrigin]
    var tex_filenames:    UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    var textures:         UnsafePointer[GpuTexture_C, MutAnyOrigin]
    var n_textures:       Int
    var shadow_tasks:     UnsafePointer[ShadowTask_C, MutAnyOrigin]
    var px_scale:         Float32
    var sobol_matrices:   UnsafePointer[UInt32, MutAnyOrigin]
    var guide:            GuideGrid
    var lights:           LightContext
    # Object instancing (see geometry.mojo's Instance_C docs). GPU call sites
    # pass dangling/zero-count values — GPU's own device-side scene upload
    # never populates instance data, so no PrimId_C.type==6 leaf can appear
    # there and these are never dereferenced on that path.
    var blasNodesArr:   UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin]
    var blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin]
    var instances:      UnsafePointer[Instance_C, MutAnyOrigin]
    # Staged spectral rendering rollout (Stage 2c, see project_spectral_rendering
    # memory) — host pointers on CPU, device pointers on GPU (see gpu.mojo's
    # ShadeContext construction sites).
    var spectral:       SpectralHandle

@always_inline
def _shading_normal(
    mesh: TriangleMesh_C,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    geo_normal: SIMD[DType.float32, 3],
    instance_idx: Int32 = Int32(-1),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
) -> SIMD[DType.float32, 3]:
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
    var n0 = SIMD[DType.float32, 3](mesh.normals[v0*3], mesh.normals[v0*3+1], mesh.normals[v0*3+2])
    var n1 = SIMD[DType.float32, 3](mesh.normals[v1*3], mesh.normals[v1*3+1], mesh.normals[v1*3+2])
    var n2 = SIMD[DType.float32, 3](mesh.normals[v2*3], mesh.normals[v2*3+1], mesh.normals[v2*3+2])
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
def _sample_level(data: UnsafePointer[Float32, MutAnyOrigin], off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
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
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
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
                    UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutAnyOrigin]](filename, su, tv, tr)
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
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
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
                    UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                    UnsafePointer[Float32, MutAnyOrigin]](filename, su, tv, tr)
                var result = RGB(_srgb_to_linear(tr[0]), _srgb_to_linear(tr[1]), _srgb_to_linear(tr[2]))
                tr.free()
                return result
    return mat.albedo


@always_inline
def shade_core(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
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

        var p0 = SIMD[DType.float32, 3](mesh.points[v0 * 4], mesh.points[v0 * 4 + 1], mesh.points[v0 * 4 + 2])
        var p1 = SIMD[DType.float32, 3](mesh.points[v1 * 4], mesh.points[v1 * 4 + 1], mesh.points[v1 * 4 + 2])
        var p2 = SIMD[DType.float32, 3](mesh.points[v2 * 4], mesh.points[v2 * 4 + 1], mesh.points[v2 * 4 + 2])

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
        var tangent = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
        var bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > 0:
            dir = dir * (1.0 / sqrt(dlen))

        # Update Ray
        var org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) + ray_dir * inter.tHit + normal * 0.0001
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
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    instance_idx: Int32 = Int32(-1),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
) -> Tuple[SIMD[DType.float32, 3], SIMD[DType.float32, 3], SIMD[DType.float32, 3]]:
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
    var rd = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    if dot(gn, rd) > Float32(0.0):
        gn = -gn
    var ro = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    return (gn, rd, ro)


@always_inline
def _shadow_contribute[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    origin: SIMD[DType.float32, 3],
    dir: SIMD[DType.float32, 3],
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
                                  ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
            path_ptr[].estimate += contrib
            # Record in the guide at the PARENT surface (one bounce back):
            # "scatter direction ray.dir from parent_cell leads to illumination W here."
            # This teaches indirect-illumination guiding — path_ptr[].ray.origin is where
            # the path came from and ray.direction is the scatter direction that arrived here.
            if Int(guide_write.energy) > 4 and path_ptr[].bounce > 0:
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
):
    var mat = ctx.materials[Int(inter.primId.materialIndex)]
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok:
        path_ptr[].active = 0
        return
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var (normal, ray_dir, ray_org) = _geom_normal_and_ray(path_ptr, p0, p1, p2, inter.primId.instanceIdx, ctx.instances)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Balance heuristic: choose reflect vs transmit proportional to luminance,
    # then a cosine-hemisphere scatter direction around the chosen lobe's normal.
    # "texture reflectance"/"texture transmittance" (e.g. a leaf.tga imagemap)
    # both resolve to the same mat.tex_idx (Material_C has one texture slot,
    # shared across kinds) — sample it once and use it for both lobes. Leaving
    # this at the flat mat.albedo/mat.emission default (as before) rendered
    # textured diffusetransmission foliage as a dull, wall-coloured grey blob
    # instead of the actual leaf texture, effectively invisible against a
    # similarly-toned wall.
    var refl = mat.albedo
    var trans = mat.emission
    if Int(mat.tex_idx) != -1:
        var tex_rgb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        refl = tex_rgb
        trans = tex_rgb
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

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok:
        path_ptr[].active = 0
        return
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures)

    var (normal, ray_dir, ray_org) = _geom_normal_and_ray(path_ptr, p0, p1, p2, inter.primId.instanceIdx, ctx.instances)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
    # Use interpolated shading normal (geometric normal still drives hit-point offset)
    normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, normal, inter.primId.instanceIdx, ctx.instances)

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
    var wo = SIMD[DType.float32, 3](-ray_dir[0], -ray_dir[1], -ray_dir[2])  # toward viewer

    # Tangent frame (Frisvad) around the shading normal for hemisphere sampling.
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
    var bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)

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
    var exit_dir = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))

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
                var env_dir: SIMD[DType.float32, 3]
                var env_rgb: RGB
                var pdf_light: Float32
                if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and Int(ilight.cdf_ptr) > 1 and ilight.cdf_w > Int32(0):
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
                exit_dir = SIMD[DType.float32, 3](wt.x, wt.y, wt.z)
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

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


# ── Shared epilogue for all delta/glossy-delta BSDFs (conductor, coated_conductor,
# dielectric, thin_dielectric): apply the sample, update throughput/bounce
# bookkeeping, run Russian roulette, and save PCG state. Each material's shade_*
# wrapper only differs in how it builds (bs, hit_point) — this is everything that
# happens once those are known. bs.is_valid is always 1 for the two dielectric
# variants, so the early-return there is a harmless no-op for them.
@always_inline
def _finish_delta_bounce(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    mut pcg: PCG32,
    bs: BxDFSample,
    hit_point: SIMD[DType.float32, 3],
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

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
    path_ptr[].pcgState = pcg.state


# ── Dielectric (glass) branch ─────────────────────────────────────────────────
@always_inline
def shade_dielectric(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        path_ptr[].active = 0
        return
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var geom_normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(geom_normal, geom_normal)
    if nlen > Float32(0.0):
        geom_normal = geom_normal * (Float32(1.0) / sqrt(nlen))

    # Smooth shading normal, oriented to the geometric/winding normal (PBRT's
    # FaceForward(Ns, Ng)): the winding — flipped at parse time by
    # ReverseOrientation — is the authoritative outside direction the dielectric
    # uses to pick entering vs exiting. (Using the raw vertex normal instead
    # breaks meshes whose vertex normals point inward, causing spurious TIR.)
    geom_normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geom_normal)

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)

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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    if not ok:
        path_ptr[].active = 0
        return
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var geom_normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(geom_normal, geom_normal)
    if nlen > Float32(0.0):
        geom_normal = geom_normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    normal: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
    alb: RGB,
    alpha: Float32,
    mat_kind: Int32,
    wo: SIMD[DType.float32, 3],
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    n: SIMD[DType.float32, 3],
    wo: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok:
        path_ptr[].active = 0
        return

    # Texture-driven roughness ("texture roughness"/"uroughness" — see
    # material_builder.mojo): resolve to a local mutable copy of `mat` here,
    # once, before anything reads roughU/V below, rather than threading a
    # texture lookup through every downstream GGX-alpha call site. Isotropic
    # only (see Material_C.rough_tex_idx's docstring); falls back to the
    # parsed scalar roughU/V when there's no texture or no UVs.
    var mat_eff = mat
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

    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var (geo_normal, ray_dir, ray_org) = _geom_normal_and_ray(path_ptr, p0, p1, p2)
    var hit_point = ray_org + ray_dir * inter.tHit + geo_normal * Float32(0.0001)
    # Use interpolated shading normal for smooth specular reflections
    var normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal)

    # roughU/V already hold the resolved GGX alpha — no squaring here.
    var alpha_x = max(mat_eff.roughU, Float32(0.0001))
    var alpha_y = max(mat_eff.roughV, Float32(0.0001))

    # Anisotropy tangent frame: UV-gradient (aligned to texture space) when the
    # mesh has UVs and the material is anisotropic; else an arbitrary Frisvad
    # frame (isotropic GGX / perfect mirror don't care about tangent direction).
    var tangent: SIMD[DType.float32, 3]
    var bitangent: SIMD[DType.float32, 3]
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
            tangent = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
            bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)
    else:
        var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
        tangent = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
        bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)

    var wo = SIMD[DType.float32, 3](-ray_dir[0], -ray_dir[1], -ray_dir[2])
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
        if path_ptr[].bounce > 1:
            var lum = path_ptr[].throughput.luma()
            var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
            if pcg.next_float() < q:
                path_ptr[].active = 0
            else:
                path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
        path_ptr[].pcgState = pcg.state
    else:
        _finish_delta_bounce(path_ptr, pcg, bs, hit_point, mat_eff.albedo)


# CoatedConductor: dielectric clearcoat over GGX conductor.
# Schlick Fresnel at the air/coat interface: F_schlick(cos_theta, 0, 1, ior)
# selects between coat specular reflection (F) and conducting GGX layer (1-F).
# This is an energy-conserving two-lobe approximation of pbrt's LayeredBxDF.
@always_inline
def shade_coated_conductor[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
):
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok:
        path_ptr[].active = 0
        return
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
    # No shading-normal interpolation here (unlike conductor) — matches the
    # pre-existing coated_conductor behavior of using the flat geometric normal.
    var (normal, ray_dir, ray_org) = _geom_normal_and_ray(path_ptr, p0, p1, p2)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var ior = mat.emission.r if mat.emission.r > Float32(1.0) else Float32(1.5)
    var wo = SIMD[DType.float32, 3](-ray_dir[0], -ray_dir[1], -ray_dir[2])
    # Frisvad frame — isotropic GGX only (single alpha), no anisotropy alignment needed.
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
    var bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)
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
        if path_ptr[].bounce > 1:
            var lum = path_ptr[].throughput.luma()
            var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
            if pcg.next_float() < q:
                path_ptr[].active = 0
            else:
                path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)
        path_ptr[].pcgState = pcg.state
    else:
        _finish_delta_bounce(path_ptr, pcg, bs, hit_point, mat.albedo)


# Mix material: randomly select one of two sub-materials using amount as probability.
# Sub-material indices are packed into mat.tex_idx: low 16 bits = idx1, high 16 bits = idx2.
# mat.roughU = blend amount (probability of picking mat2).
# Not @always_inline: shade_mix ↔ _shade_dispatch would form an always_inline recursion.
def shade_mix[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
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
    geom_normal: SIMD[DType.float32, 3],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    pixel_uv: Float32 = Float32(0.0),
) -> SIMD[DType.float32, 3]:
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    mat: Material_C,
    ctx: ShadeContext,
) -> Tuple[GeomContext, Bool]:
    var (mesh, v0, v1, v2, ok) = _get_tri_verts(inter, ctx.meshes)
    if not ok:
        var z = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
        return (GeomContext(z, z, z, z, z, z, RGB(Float32(0.0)), Float32(0.0)), False)
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
    var (geo_normal, ray_dir, ray_org) = _geom_normal_and_ray(path_ptr, p0, p1, p2, inter.primId.instanceIdx, ctx.instances)
    var ng_ff = geo_normal

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
    var wo = SIMD[DType.float32, 3](-ray_dir[0], -ray_dir[1], -ray_dir[2])
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
    var bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)
    return (GeomContext(normal, ng_ff, hit_point, wo, tangent, bitangent, alb, pixel_uv), True)

# Extract 8 consecutive Sobol dimensions for one non-delta bounce and advance
# the path's sampler_dim counter.
@always_inline
def _draw_sobol_8(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
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
    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var wo = -ray_dir
    var hc = _hair_precompute(mat, ctx.curves, curve_idx, v_global, h, wo)
    # NEE (all light types) and indirect sampling now go through hc directly
    # (bxdf.mojo's _nee_weight_hair, bvh.mojo's _hair_sample_dir) — only
    # geo_normal is still needed locally, for shadow-ray/hit-point offsets.
    var geo_normal = hc.geo_normal

    # ── Hit point for shadow rays (always offset toward incoming side) ────────
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
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

    # Russian roulette
    if path_ptr[].bounce > 1:
        var lum_t = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum_t if lum_t < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


@always_inline
def _mnee_vertex_mats(
    wi: SIMD[DType.float32, 3], wo: SIMD[DType.float32, 3],
    H: SIMD[DType.float32, 3], s: SIMD[DType.float32, 3], t: SIMD[DType.float32, 3],
    dp_du: SIMD[DType.float32, 3], dp_dv: SIMD[DType.float32, 3],
    ili: Float32, ilo: Float32,
    dp_du_prev: SIMD[DType.float32, 3], dp_dv_prev: SIMD[DType.float32, 3],
    dp_du_next: SIMD[DType.float32, 3], dp_dv_next: SIMD[DType.float32, 3],
    has_prev: Bool, has_next: Bool,
) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4], SIMD[DType.float32, 4], SIMD[DType.float32, 2]]:
    # Computes (a, b, c, constraint) matrices at a specular vertex.
    # a=coupling to prev, b=self, c=coupling to next; uses Cycles mnee.h formulas.
    var b_du = -(dp_du*(ili+ilo)) + wi*(dot(wi,dp_du)*ili) + wo*(dot(wo,dp_du)*ilo)
    var b_dv = -(dp_dv*(ili+ilo)) + wi*(dot(wi,dp_dv)*ili) + wo*(dot(wo,dp_dv)*ilo)
    b_du -= H*dot(b_du,H); b_du = -b_du
    b_dv -= H*dot(b_dv,H); b_dv = -b_dv
    var b = SIMD[DType.float32, 4](dot(b_du,s), dot(b_dv,s), dot(b_du,t), dot(b_dv,t))
    var a = SIMD[DType.float32, 4](Float32(0))
    if has_prev:
        var a_du = (dp_du_prev - wi*dot(wi,dp_du_prev)) * ili
        var a_dv = (dp_dv_prev - wi*dot(wi,dp_dv_prev)) * ili
        a_du -= H*dot(a_du,H); a_du = -a_du
        a_dv -= H*dot(a_dv,H); a_dv = -a_dv
        a = SIMD[DType.float32, 4](dot(a_du,s), dot(a_dv,s), dot(a_du,t), dot(a_dv,t))
    var c = SIMD[DType.float32, 4](Float32(0))
    if has_next:
        var c_du = (dp_du_next - wo*dot(wo,dp_du_next)) * ilo
        var c_dv = (dp_dv_next - wo*dot(wo,dp_dv_next)) * ilo
        c_du -= H*dot(c_du,H); c_du = -c_du
        c_dv -= H*dot(c_dv,H); c_dv = -c_dv
        c = SIMD[DType.float32, 4](dot(c_du,s), dot(c_dv,s), dot(c_du,t), dot(c_dv,t))
    var constraint = SIMD[DType.float32, 2](dot(H,s), dot(H,t))
    return (a, b, c, constraint)

@always_inline
def _mat22_mul(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3], a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3])

@always_inline
def _mat22_mul_v(m: SIMD[DType.float32, 4], v: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](m[0]*v[0]+m[1]*v[1], m[2]*v[0]+m[3]*v[1])

@always_inline
def _mat22_inv(m: SIMD[DType.float32, 4]) -> Tuple[SIMD[DType.float32, 4], Float32]:
    var det = m[0]*m[3] - m[1]*m[2]
    if abs(det) < Float32(1e-5):
        return (SIMD[DType.float32, 4](Float32(0)), Float32(0))
    return (SIMD[DType.float32, 4](m[3], -m[1], -m[2], m[0]) * (Float32(1.0)/det), det)

# 2-vertex MNEE: Newton walk for x0→x1(glass1)→x2(glass2)→x3.
# etas must be pre-computed: eta = ior if entering glass, 1/ior if exiting.
# Returns (converged, x1_f, x2_f, bsdf_product, dx1_dxlight).
@always_inline
def _mnee_walk2(
    x0: SIMD[DType.float32, 3], x3: SIMD[DType.float32, 3],
    x1_init: SIMD[DType.float32, 3], n1: SIMD[DType.float32, 3],
    dp_du1: SIMD[DType.float32, 3], dp_dv1: SIMD[DType.float32, 3], eta1: Float32,
    x2_init: SIMD[DType.float32, 3], n2: SIMD[DType.float32, 3],
    dp_du2: SIMD[DType.float32, 3], dp_dv2: SIMD[DType.float32, 3], eta2: Float32,
    ldp_du: SIMD[DType.float32, 3], ldp_dv: SIMD[DType.float32, 3],
) -> Tuple[Bool, SIMD[DType.float32, 3], SIMD[DType.float32, 3], Float32, Float32]:
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
        var (a0,b0,c0,cv0) = _mnee_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
            SIMD[DType.float32,3](0),SIMD[DType.float32,3](0),dp_du2,dp_dv2,False,True)
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
        var (a1,b1,c1,cv1) = _mnee_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
            dp_du1,dp_dv1,SIMD[DType.float32,3](0),SIMD[DType.float32,3](0),True,False)
        _ = a0; _ = c1
        # ── Convergence ──────────────────────────────────────────────────────
        var err = max(max(abs(cv0[0]),abs(cv0[1])), max(abs(cv1[0]),abs(cv1[1])))
        if err < Float32(1e-3):
            converged = True; break
        # ── Block tridiagonal solve: [b0 c0; a1 b1]*[dx0;dx1]=[cv0;cv1] ────
        var (Li0, det0) = _mat22_inv(b0)
        if det0 == Float32(0): break
        var A1 = _mat22_mul(a1, Li0)
        var Lk1 = b1 - _mat22_mul(A1, c0)
        var (Li1, det1) = _mat22_inv(Lk1)
        if det1 == Float32(0): break
        var C1_red = cv1 - _mat22_mul_v(A1, cv0)
        var dx1 = _mat22_mul_v(Li1, C1_red)
        var dx0 = _mat22_mul_v(Li0, cv0 - _mat22_mul_v(c0, dx1))
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
    var (a0f,b0f,c0f,cv0f) = _mnee_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
        SIMD[DType.float32,3](0),SIMD[DType.float32,3](0),dp_du2,dp_dv2,False,True)
    var (a1f,b1f,c1f,cv1f) = _mnee_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
        dp_du1,dp_dv1,SIMD[DType.float32,3](0),SIMD[DType.float32,3](0),True,False)
    _ = a0f; _ = c1f; _ = cv0f; _ = cv1f
    # ── Block tridiagonal LU factorization for transfer matrix ───────────────
    var (Li0f, det0f) = _mat22_inv(b0f)
    if det0f == Float32(0): return (False, x1_init, x2_init, Float32(0), Float32(0))
    var U0 = _mat22_mul(Li0f, c0f)
    var Lk1f = b1f - _mat22_mul(a1f, U0)
    var (Li1f, det1f) = _mat22_inv(Lk1f)
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
    var cosNI1 = abs(dot(n1,wi1)); var cosHI1 = abs(dot(H1,wi1)); var cosTM1 = abs(dot(n1,H1))
    var F1 = fr_dielectric(cosNI1, eta1); var bsdf1 = (Float32(1)-F1)*cosHI1/max(cosNI1*cosTM1*cosTM1,Float32(1e-6))
    var cosNI2 = abs(dot(n2,wi2)); var cosHI2 = abs(dot(H2,wi2)); var cosTM2 = abs(dot(n2,H2))
    var F2 = fr_dielectric(cosNI2, eta2); var bsdf2 = (Float32(1)-F2)*cosHI2/max(cosNI2*cosTM2*cosTM2,Float32(1e-6))
    return (True, x1, x2, bsdf1*bsdf2, dx1_dxlight)

@always_inline
# MNEE manifold walk: Newton iteration to find x1 on a glass surface such that
# Snell's law holds for the path x0 → x1 → x2. Works for flat glass surfaces
# (no reprojection needed). Returns (converged, x1_final, det_b, eta_final).
# det_b is the 2x2 constraint Jacobian determinant needed for the transfer matrix.
@always_inline
def _mnee_walk(
    x0: SIMD[DType.float32, 3],
    x2: SIMD[DType.float32, 3],
    x1_init: SIMD[DType.float32, 3],
    n_in: SIMD[DType.float32, 3],
    dp_du: SIMD[DType.float32, 3],
    dp_dv: SIMD[DType.float32, 3],
    eta_in: Float32,
) -> Tuple[Bool, SIMD[DType.float32, 3], Float32, Float32]:
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
        if max(abs(cs), abs(ct)) < Float32(1e-3):
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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    ilight: InfiniteLight_C,
    normal: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
    alb: RGB,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid,
):
    var env_dir: SIMD[DType.float32, 3]
    var env_rgb: RGB
    var pdf_light: Float32

    if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and Int(ilight.cdf_ptr) > 1 and ilight.cdf_w > Int32(0):
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
) -> Tuple[SIMD[DType.float32, 3], SIMD[DType.float32, 3], SIMD[DType.float32, 3], SIMD[DType.float32, 3]]:
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
        var axis_dir = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
        if axis_len > Float32(1e-8):
            axis_dir = axis * (Float32(1.0) / axis_len)
        var r = r0 + (r1 - r0) * u1
        var u_perp = _curve_perp_axis(axis_dir)
        var v_perp = cross(axis_dir, u_perp)
        var theta = u2 * TWO_PI
        var radial = u_perp * cos(theta) + v_perp * sin(theta)
        var point = q0 + axis_dir * (axis_len * u1) + radial * r
        var zero3 = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
        return (point, radial, zero3, zero3)
    else:
        var lmesh = ctx.meshes[Int(al.meshIdx)]
        var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
        var lb = lti * 3
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
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
    hit_point: SIMD[DType.float32, 3],
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
    var invalid = LightSample(SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0)),
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
def _nee_area_lights[enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    normal: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
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
                               ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances)
            var probe_inter = probe_store[0]
            var used_mnee = False
            # MNEE's glass-refraction focusing needs the light's own surface
            # tangents (ldp_du_v/ldp_dv_v below) — well-defined for a flat
            # mesh triangle, not for a curve's swept tube. Curve lights fall
            # back to plain (non-MNEE) shadow-ray NEE through glass.
            if al.kind == Int8(0) and probe_inter.hit != Int8(0) and probe_inter.primId.type == Int8(0):
                var probe_mat = ctx.materials[Int(probe_inter.primId.materialIndex)]
                if probe_mat.type == MatKind.dielectric or probe_mat.type == MatKind.thin_dielectric:
                    # --- Extract x1 geometry ---
                    var (pmesh, pv0, pv1, pv2, _) = _get_tri_verts(probe_inter, ctx.meshes)
                    used_mnee = True
                    var pp0 = SIMD[DType.float32, 3](pmesh.points[pv0*4], pmesh.points[pv0*4+1], pmesh.points[pv0*4+2])
                    var pp1 = SIMD[DType.float32, 3](pmesh.points[pv1*4], pmesh.points[pv1*4+1], pmesh.points[pv1*4+2])
                    var pp2 = SIMD[DType.float32, 3](pmesh.points[pv2*4], pmesh.points[pv2*4+1], pmesh.points[pv2*4+2])
                    var pdp_du = pp1 - pp0
                    var pdp_dv = pp2 - pp0
                    var pgeo_n3 = cross(pdp_du, pdp_dv)
                    var pgeo_n_len = sqrt(dot(pgeo_n3, pgeo_n3))
                    if pgeo_n_len > Float32(1e-10):
                        var pgeo_n_raw = pgeo_n3 * (Float32(1.0) / pgeo_n_len)
                        # eta1: entering if probe goes against raw normal
                        var ior1 = probe_mat.albedo.r
                        var eta1 = ior1 if dot(pgeo_n_raw, shadow_dir) <= Float32(0) else (Float32(1.0)/ior1)
                        var pgeo_n = pgeo_n_raw
                        if dot(pgeo_n, shadow_dir) > Float32(0.0):
                            pgeo_n = -pgeo_n
                        var pu = probe_inter.u; var pvb = probe_inter.v
                        var x1_init = pp0*(Float32(1.0)-pu-pvb) + pp1*pu + pp2*pvb
                        # --- Probe for second glass surface beyond x1 ---
                        var probe2_t0 = probe_inter.tHit
                        var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
                        var probe2_org = x1_init + shadow_dir * Float32(0.0005)
                        var probe2_inter = dummy_inter
                        if probe2_rem > Float32(0.001):
                            var probe2_ray = Ray_C(
                                Point3f(probe2_org[0], probe2_org[1], probe2_org[2]),
                                Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
                            var probe2_store = InlineArray[Intersection_C, 1](fill=dummy_inter)
                            traverse_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                                       ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances)
                            probe2_inter = probe2_store[0]
                        if probe2_inter.hit != Int8(0) and probe2_inter.primId.type == Int8(0):
                            var probe2_mat = ctx.materials[Int(probe2_inter.primId.materialIndex)]
                            if probe2_mat.type == MatKind.dielectric or probe2_mat.type == MatKind.thin_dielectric:
                                # --- 2-vertex MNEE ---
                                var (p2mesh, p2v0, p2v1, p2v2, _) = _get_tri_verts(probe2_inter, ctx.meshes)
                                var p2p0 = SIMD[DType.float32, 3](p2mesh.points[p2v0*4], p2mesh.points[p2v0*4+1], p2mesh.points[p2v0*4+2])
                                var p2p1 = SIMD[DType.float32, 3](p2mesh.points[p2v1*4], p2mesh.points[p2v1*4+1], p2mesh.points[p2v1*4+2])
                                var p2p2 = SIMD[DType.float32, 3](p2mesh.points[p2v2*4], p2mesh.points[p2v2*4+1], p2mesh.points[p2v2*4+2])
                                var pdp_du2 = p2p1 - p2p0; var pdp_dv2 = p2p2 - p2p0
                                var pgeo_n3_2 = cross(pdp_du2, pdp_dv2)
                                var pgeo_n_len2 = sqrt(dot(pgeo_n3_2, pgeo_n3_2))
                                if pgeo_n_len2 > Float32(1e-10):
                                    var pgeo_n2_raw = pgeo_n3_2 * (Float32(1.0)/pgeo_n_len2)
                                    var ior2 = probe2_mat.albedo.r
                                    var eta2 = ior2 if dot(pgeo_n2_raw, shadow_dir) <= Float32(0) else (Float32(1.0)/ior2)
                                    var pgeo_n2 = pgeo_n2_raw
                                    if dot(pgeo_n2, shadow_dir) > Float32(0.0):
                                        pgeo_n2 = -pgeo_n2
                                    var pu2 = probe2_inter.u; var pvb2 = probe2_inter.v
                                    var x2_init = p2p0*(Float32(1.0)-pu2-pvb2) + p2p1*pu2 + p2p2*pvb2
                                    var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
                                        hit_point, light_point,
                                        x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
                                        x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
                                        ldp_du_v, ldp_dv_v)
                                    if ok2:
                                        var wi2f = hit_point - x1_f2
                                        var wi2fl = sqrt(dot(wi2f,wi2f))
                                        if wi2fl > Float32(1e-8):
                                            var wi2fn = wi2f*(Float32(1)/wi2fl)
                                            var cos_s_x0 = dot(normal, -wi2fn)
                                            if cos_s_x0 > Float32(0):
                                                var G2 = min(abs(dot(wi2fn,pgeo_n))/(wi2fl*wi2fl)*dx1_dxl2, Float32(2.0))
                                                var pdf_area2 = light_sel_pdf_nee / al.total_area
                                                var wo2f = light_point - x2_f2
                                                var wo2fl = sqrt(dot(wo2f,wo2f))
                                                if wo2fl > Float32(1e-8):
                                                    var wo2fn = wo2f*(Float32(1)/wo2fl)
                                                    var vis2_org = x2_f2 + wo2fn*Float32(0.001)
                                                    var vis2_ray = Ray_C(Point3f(vis2_org[0],vis2_org[1],vis2_org[2]), Vec3f(wo2fn[0],wo2fn[1],wo2fn[2]))
                                                    if not any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis2_ray, wo2fl*Float32(0.999),
                                                                  ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
                                                        var mnee_wt2 = bxdf_eval_diffuse(alb) * al.emission * (cos_s_x0 * G2 * bsdf_prod * lobe_w / pdf_area2)
                                                        path_ptr[].estimate += path_ptr[].throughput * mnee_wt2
                        else:
                            # --- 1-vertex MNEE ---
                            var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(
                                hit_point, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
                            if mnee_ok:
                                var wi_f = hit_point - x1_f
                                var wi_len2_f = dot(wi_f, wi_f)
                                var wo_f = light_point - x1_f
                                var wo_len2_f = dot(wo_f, wo_f)
                                if wi_len2_f > Float32(1e-8) and wo_len2_f > Float32(1e-8):
                                    var wi_len_f = sqrt(wi_len2_f)
                                    var wo_len_f = sqrt(wo_len2_f)
                                    var wi_fn = wi_f * (Float32(1.0) / wi_len_f)
                                    var wo_fn = wo_f * (Float32(1.0) / wo_len_f)
                                    var cos_s_x0 = dot(normal, -wi_fn)
                                    if cos_s_x0 > Float32(0.0):
                                        var H3_f = -(wi_fn + wo_fn * eta_f)
                                        var H_len2_f = dot(H3_f, H3_f)
                                        if H_len2_f > Float32(1e-10):
                                            var H_len_f = sqrt(H_len2_f)
                                            var H_f = H3_f * (Float32(1.0) / H_len_f)
                                            var dp_du_dot_n = dot(pdp_du, pgeo_n)
                                            var s3_f = pdp_du - pgeo_n * dp_du_dot_n
                                            var s_len2_f = dot(s3_f, s3_f)
                                            if s_len2_f > Float32(1e-10):
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
                                                var dw0_dx1 = abs(dot(wi_fn, pgeo_n)) / (wi_len2_f)
                                                var G = min(dw0_dx1 * dx1_dxl, Float32(2.0))
                                                var cosNI = abs(dot(pgeo_n, wi_fn))
                                                var cosHI = abs(dot(H_f, wi_fn))
                                                var cosTM = abs(dot(pgeo_n, H_f))
                                                var F_r = fr_dielectric(cosNI, eta_f)
                                                var T_f = Float32(1.0) - F_r
                                                var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6))
                                                var pdf_area_x2 = light_sel_pdf_nee / al.total_area
                                                var vis_org = x1_f + wo_fn * Float32(0.001)
                                                var vis_ray = Ray_C(
                                                    Point3f(vis_org[0], vis_org[1], vis_org[2]),
                                                    Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
                                                if not any_hit_bvh2_core(ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis_ray, wo_len_f * Float32(0.999),
                                                              ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
                                                    var mnee_wt = bxdf_eval_diffuse(alb) * al.emission * (cos_s_x0 * G * bsdf_s * lobe_w / pdf_area_x2)
                                                    path_ptr[].estimate += path_ptr[].throughput * mnee_wt
            if not used_mnee:
                _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, shadow_dir, dist * Float32(0.9999), contrib, guide_write)

def _shade_diffuse_nee[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    ctx: ShadeContext,
    normal: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
    alb: RGB,
    wo: SIMD[DType.float32, 3],
    u_light: Float32,
    u_bary1: Float32,
    u_bary2: Float32,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid = null_guide(),
):
    # NEE sampling asymmetry: area lights use CDF-weighted selection (one light per
    # bounce, weight = power), while infinite/env lights are ALL sampled every bounce.
    # Rationale: area lights are finite and numerous (N can be large), so stochastic
    # selection with MIS is necessary. Infinite lights are typically 1-2 env maps,
    # and their contribution is often dominant — sampling all of them every bounce
    # costs little and avoids the variance of single-sample env selection. At N>2
    # env lights this asymmetry would need revisiting.

    # ── Area light NEE ────────────────────────────────────────────────────────
    _nee_area_lights[enqueue_shadow](path_ptr, ctx, normal, hit_point, alb, u_light, u_bary1, u_bary2, pcg, guide_write)

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
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    mat: Material_C,
    guide_write: GuideGrid = null_guide(),
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
        u_light, u_bary1, u_bary2, u_env1, u_env2, pcg, guide_write)

    # ── Scatter direction: 50/50 mixture of guide and cosine-weighted BSDF ──────
    # The guide is active when its energy pointer is a real allocation (Int > 1).
    # MIS balance heuristic: weight = f·cosθ / pdf_mix, where
    #   pdf_mix = 0.5·pdf_guide + 0.5·pdf_bsdf  and  f = alb/π (Lambertian).
    var dir: SIMD[DType.float32, 3]
    var cos_theta: Float32
    var pdf_mix: Float32

    if Int(ctx.guide.energy) > 4:
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
                dir = SIMD[DType.float32, 3](gdx, gdy, gdz)
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

    if path_ptr[].bounce > 1:
        var lum = path_ptr[].throughput.luma()
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if u_rr < q:
            path_ptr[].active = 0
        else:
            path_ptr[].throughput *= Float32(1.0) / (Float32(1.0) - q)

    path_ptr[].pcgState = pcg.state


@always_inline
def shade_interface(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
):
    """Interface (null/passthrough) material: advance the ray through the surface.
    No scattering, no throughput change. Medium transition is handled externally:
    on CPU by rendering.mojo's medium-interface loop; on GPU by shade_interface_gpu."""
    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + ray_dir * Float32(0.0002)
    path_ptr[].ray = Ray_C(Point3f(hit_point[0], hit_point[1], hit_point[2]), path_ptr[].ray.direction)
    path_ptr[].specularBounce = Int8(1)
    path_ptr[].lastBsdfPdf = Float32(0.0)


@always_inline
def _shade_dispatch[use_gpu: Bool, enqueue_shadow: Bool](
    mat: Material_C,
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
):
    if mat.type == MatKind.diffuse:
        shade_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat, guide_write)
    # Delta BSDFs (dielectric variants) need only triangle geometry — no NEE,
    # textures, or Sobol. Passing ctx.meshes directly keeps GPU kernel
    # argument counts minimal. Conductor/coated_conductor CAN be rough (not
    # delta), so they now take full ctx for NEE — see shade_conductor's
    # own docstring/_shade_conductor_nee.
    elif mat.type == MatKind.conductor:
        shade_conductor[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.dielectric:
        shade_dielectric(path_ptr, inter, ctx.meshes, mat)
    elif mat.type == MatKind.coated_diffuse:
        shade_coated_diffuse[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.diffuse_transmit:
        shade_diffuse_transmission[use_gpu, enqueue_shadow](path_ptr, inter, ctx)
    elif mat.type == MatKind.coated_conductor:
        shade_coated_conductor[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.mix:
        shade_mix[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    elif mat.type == MatKind.thin_dielectric:
        shade_thin_dielectric(path_ptr, inter, ctx.meshes, mat)
    elif mat.type == MatKind.interface:
        shade_interface(path_ptr, inter)
    elif mat.type == MatKind.hair:
        comptime if not use_gpu:
            shade_hair[use_gpu, enqueue_shadow](path_ptr, inter, ctx, mat)
    else:
        path_ptr[].active = 0


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup uses OIIO external_call on CPU and device-resident GpuTexture_C on GPU.
@always_inline
def shade_nee_core[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    ctx: ShadeContext,
    guide_write: GuideGrid = null_guide(),
):
    # ── Miss handler: ray escaped — add infinite light and deactivate ──────────
    if inter.hit == 0:
        # Volume scatter sets hit=0 to skip surface shading but is not a true miss.
        if path_ptr[].volume_scattered == Int8(1):
            path_ptr[].volume_scattered = Int8(0)
            return
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
        var miss_albedo = RGB(Float32(0.0))
        for inf_i in range(ctx.lights.infinite_count):
            var ilight = ctx.lights.infinite_lights[inf_i]
            # Transform world-space ray direction into light's local frame
            var w2l = ilight.world_to_light
            var ld_x = w2l[0]*ray_dir[0] + w2l[4]*ray_dir[1] + w2l[8]*ray_dir[2]
            var ld_y = w2l[1]*ray_dir[0] + w2l[5]*ray_dir[1] + w2l[9]*ray_dir[2]
            var ld_z = w2l[2]*ray_dir[0] + w2l[6]*ray_dir[1] + w2l[10]*ray_dir[2]
            var local_dir = SIMD[DType.float32, 3](ld_x, ld_y, ld_z)
            var env_rgb: RGB
            if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 4 and ilight.cdf_w > Int32(0):
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
                            UnsafePointer[UInt8, MutAnyOrigin], Float32, Float32,
                            UnsafePointer[Float32, MutAnyOrigin]](fname, u, v, tr)
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
                        if Int(ilight.pixels_ptr) > 1:
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
                var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
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
        if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
            path_ptr[].estimate += path_ptr[].throughput * emission
        else:
            var pdf_bsdf = path_ptr[].lastBsdfPdf
            if pdf_bsdf > Float32(0.0):
                var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(inter, ctx.meshes)
                var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
                var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
                var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
                var lnorm = cross(lp1 - lp0, lp2 - lp0)
                var lnlen = dot(lnorm, lnorm)
                if lnlen > Float32(0.0):
                    lnorm = lnorm * (Float32(1.0) / sqrt(lnlen))
                var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
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
                        var tangent = SIMD[DType.float32, 3](Float32(1.0), Float32(0.0), Float32(0.0))
                        if seg_len > Float32(1e-8):
                            tangent = seg_axis * (Float32(1.0) / seg_len)
                        var n_perp = _curve_perp_axis(tangent)
                        var b_perp0 = cross(tangent, n_perp)
                        var geo_normal = n_perp * h + b_perp0 * sqrt(max(Float32(0.0), Float32(1.0) - h*h))
                        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
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
        _shade_dispatch[False, enqueue_shadow](mat, path_ptr, inter, ctx, guide_write)


@always_inline
def shade_core_cpu_nee(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    tid: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int,
    light_sampler: LightSampler_C,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    guide: GuideGrid,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
    guide_write: GuideGrid = null_guide(),
    spectral: SpectralHandle = null_spectral_handle(),
):
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=tex_filenames,
        textures=UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), n_textures=0,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=Float32(0.0), sobol_matrices=sobol_matrices, guide=guide,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=spectral,
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=distantLightCount,
            point_lights=pointLights, point_count=pointLightCount,
            infinite_lights=infiniteLights, infinite_count=infiniteLightCount,
            spheres=spheres, sphere_count=sphereCount, light_sampler=light_sampler))
    shade_nee_core[False, False](path_ptr, inter, ctx, guide_write)
