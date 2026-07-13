from std.memory import alloc
from std.math import sqrt, cos, sin, max, min, exp, floor, log
from std.algorithm import parallelize
from .geometry import Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, Curve_C, intersect_curve, CURVE_DEFER_K, CURVE_N_PIECES, curve_piece_endpoints, _curve_perp_axis, DistantLight_C, PointLight_C, InfiniteLight_C, dot, cross, intersect_triangle, PathState_C, TileResult_C, Point3f, Point2f, Vec3f, Frame, RGB, Medium_C, MediumInterface_C, Grid_C, LightSampler_C, Instance_C, PI, TWO_PI, INV_PI, INV_FOUR_PI, safe_sqrt, fr_dielectric, sphere_outward_normal, MeasuredBRDF_C, GpuTexture_C, _is_real_ptr
from .rng import PCG32
from .spectrum import SpectralHandle

# ── BVH2 Compact Nodes (32 bytes per node, 1 cache line) ──────────────────────
# Layout: Point3f min (12 B) + Point3f max (12 B) + Int32 offset (4 B) + Int32 count (4 B) = 32 B

@fieldwise_init
struct BVH2Node(TrivialRegisterPassable):
    var min: Point3f        # AABB minimum corner
    var max: Point3f        # AABB maximum corner
    var offset: Int32       # interior: right child index, leaf: primIds offset
    var count: Int32        # 0 = interior, >0 = leaf primitive count

@always_inline
def intersect_aabb(
    bmin: Point3f, bmax: Point3f,
    rdir: Vec3f,
    org: Vec3f,
    nearXIsMin: Bool, nearYIsMin: Bool, nearZIsMin: Bool,
    tMax: Float32
) -> Tuple[Bool, Float32]:
    var nearX = bmin.x if nearXIsMin else bmax.x
    var farX  = bmax.x if nearXIsMin else bmin.x
    var nearY = bmin.y if nearYIsMin else bmax.y
    var farY  = bmax.y if nearYIsMin else bmin.y
    var nearZ = bmin.z if nearZIsMin else bmax.z
    var farZ  = bmax.z if nearZIsMin else bmin.z

    # (near - org) * rdir, NOT near*rdir - org*rdir: algebraically identical
    # for finite rdir, but for an axis-aligned ray (rdir.x = +/-inf) the two
    # separately-computed infinite products in the old form were routinely
    # like-signed, so subtracting them produced inf-inf = NaN and a false
    # miss even for rays that truly passed through the box. Subtracting the
    # (finite) coordinates FIRST and only then multiplying by the infinite
    # rdir keeps the arithmetic finite until the last step, so it only
    # degenerates to 0*inf=NaN in the genuinely ambiguous case of the ray
    # origin sitting exactly on the box face — same per-node op count either
    # way (one subtract + one multiply per axis), so this costs nothing.
    var tNearX = (nearX - org.x) * rdir.x
    var tNearY = (nearY - org.y) * rdir.y
    var tNearZ = (nearZ - org.z) * rdir.z

    var tFarX = (farX - org.x) * rdir.x
    var tFarY = (farY - org.y) * rdir.y
    var tFarZ = (farZ - org.z) * rdir.z

    # tNear = max(tNearX, tNearY, tNearZ, 0)
    var tNear = tNearX if tNearX > tNearY else tNearY
    tNear = tNearZ if tNearZ > tNear else tNear
    tNear = Float32(0.0) if Float32(0.0) > tNear else tNear

    # tFar = min(tFarX, tFarY, tFarZ, tMax) * gamma
    var tFar = tFarX if tFarX < tFarY else tFarY
    tFar = tFarZ if tFarZ < tFar else tFar
    tFar = tMax if tMax < tFar else tFar
    tFar = tFar * Float32(1.0000003)

    return (tNear <= tFar, tNear)

@fieldwise_init
struct SceneDescriptor2_C(TrivialRegisterPassable):
    var bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin]
    var primIds: UnsafePointer[PrimId_C, MutAnyOrigin]
    var meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin]
    var meshCount: Int64
    var materials: UnsafePointer[Material_C, MutAnyOrigin]
    var materialCount: Int64
    var areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin]
    var areaLightCount: Int64
    var textures: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    var textureCount: Int64
    var distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin]
    var distantLightCount: Int64
    var pointLights: UnsafePointer[PointLight_C, MutAnyOrigin]
    var pointLightCount: Int64
    var infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin]
    var infiniteLightCount: Int64
    var spheres: UnsafePointer[Sphere_C, MutAnyOrigin]
    var sphereCount: Int64
    var curves: UnsafePointer[Curve_C, MutAnyOrigin]
    var curveCount: Int64
    var mediums: UnsafePointer[Medium_C, MutAnyOrigin]
    var mediumCount: Int64
    var mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin]
    var mediumIfaceCount: Int64
    var grids: UnsafePointer[Grid_C, MutAnyOrigin]
    var gridCount: Int64
    var lightSampler: LightSampler_C

    # Object instancing: one private BVH2 ("BLAS") per template, each a
    # separate allocation reachable via Instance_C.blasIdx, plus TLAS instance
    # placements. See geometry.mojo's Instance_C docs. instanceCount == 0 for
    # scenes with no ObjectInstance usage (the blas*/instances pointers may
    # then be dangling — never dereferenced since no PrimId_C.type==6 leaf
    # exists in that case).
    var blasNodesArr:   UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin]
    var blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin]
    var blasCount:      Int64
    var instances:      UnsafePointer[Instance_C, MutAnyOrigin]
    var instanceCount:  Int64

    # "measured" materials: one MeasuredBRDF_C per distinct .bsdf file
    # (deduped at scene-build time), referenced by Material_C.measured_idx.
    # See measured_bsdf.mojo's loader / pbrt_parser.mojo's finalize_scene.
    var measuredBrdfs:      UnsafePointer[MeasuredBRDF_C, MutAnyOrigin]
    var measuredBrdfCount:  Int64

    # Staged spectral rendering rollout (see project_spectral_rendering memory
    # / lovely-dazzling-meteor plan, Stage 2c). Loaded once per render and
    # threaded through here so render_tile/shade_core_cpu_nee need no new
    # parameter of their own — they already carry a SceneDescriptor2_C.
    # Dangling (null_spectral_handle()) for BDPT/SPPM (_mk_sd_full below,
    # Stage 3/4 — not wired yet) and test fixtures that don't supply a real
    # loaded table.
    var spectral: SpectralHandle

    # GPU-resident image textures (one GpuTexture_C per distinct file,
    # mip-chained device buffers — see gpu.mojo's GpuSceneHandle.textures_buf,
    # the only real producer of this array). `textures`/`textureCount` above
    # are the CPU-only tex_filenames path (OIIO bridge lookups); this is the
    # separate GPU-only path shading.mojo's _tex_lookup[True] branch needs.
    # Dangling/0 for BDPT/SPPM CPU callers (scene.mojo's scene_descriptor(),
    # which has no GPU buffers to offer) and any caller that never reaches a
    # material with an image-texture reflectance.
    var gpuTextures: UnsafePointer[GpuTexture_C, MutAnyOrigin]
    var gpuTextureCount: Int64

@always_inline
def _mk_sd_full(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    meshCount: Int64,
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    materialCount: Int64,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin] = UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(),
    distantLightCount: Int64 = Int64(0),
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin] = UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(),
    infiniteLightCount: Int64 = Int64(0),
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin] = UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(),
    pointLightCount: Int64 = Int64(0),
    # Staged spectral rendering rollout, Stage 3 (BDPT). Decomposed into
    # individual pointer/int params -- NOT a single `spectral: SpectralHandle`
    # by-value param -- because that shape is a confirmed, reproducible Mojo
    # miscompilation (see spectrum.mojo's comment above rgb_to_spectral_sample
    # / project_spectral_rendering memory). Defaults match null_spectral_handle()
    # so existing (SPPM, Stage 4) callers that don't pass these are unaffected.
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfs: UnsafePointer[MeasuredBRDF_C, MutAnyOrigin] = UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
    measuredBrdfCount: Int64 = Int64(0),
    gpuTextures: UnsafePointer[GpuTexture_C, MutAnyOrigin] = UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(),
    gpuTextureCount: Int64 = Int64(0),
) -> SceneDescriptor2_C:
    """Builds a complete SceneDescriptor2_C from raw GPU device pointers so
    the SAME `sd.field`-based traversal code a CPU-side function already
    uses works unmodified on GPU (SceneDescriptor2_C is
    TrivialRegisterPassable — cheap to construct per-thread, no allocation).
    Lives here (not in bdpt.mojo, where it originated, or sppm.mojo, which
    also needs it) specifically to avoid an import cycle — both of those
    modules import shared helpers from each other already, but neither
    imports from the other, so a shared GPU-scene-builder helper needs a
    home neither of them owns; this is that home, next to
    SceneDescriptor2_C itself. Only the fields bdpt.mojo/sppm.mojo's shared
    functions actually dereference are filled from real device buffers;
    grids and the light sampler CDF are never touched by either module's
    code paths, so they stay dangling/zero-count, same convention
    traverse_bvh2_core's own optional instancing args already use.
    distant/infinite/point lights default to the same dangling convention
    for backward compatibility, but callers that need distant/infinite/
    point-light NEE or light-path emission (both bdpt.mojo and sppm.mojo now
    do, see [[project_priority_backlog]] item 1) should pass the real device
    buffers/counts explicitly. `gpuTextures`/`gpuTextureCount` likewise
    default to dangling/0 -- callers that need real image-texture
    reflectance (bdpt.mojo's GPU kernels, since c9f7e20-era coateddiffuse/
    diffuse materials with "texture reflectance" silently fell back to a
    flat 0.5 grey default otherwise) should pass GpuSceneHandle's own
    textures_buf/n_textures explicitly."""
    return SceneDescriptor2_C(
        bvh2Nodes=bvh2Nodes, primIds=primIds,
        meshes=meshes, meshCount=meshCount,
        materials=materials, materialCount=materialCount,
        areaLights=areaLights, areaLightCount=areaLightCount,
        textures=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textureCount=Int64(0),
        distantLights=distantLights,
        distantLightCount=distantLightCount,
        pointLights=pointLights,
        pointLightCount=pointLightCount,
        infiniteLights=infiniteLights,
        infiniteLightCount=infiniteLightCount,
        spheres=spheres, sphereCount=sphereCount,
        curves=curves, curveCount=curveCount,
        mediums=mediums, mediumCount=mediumCount,
        mediumInterfaces=mediumInterfaces, mediumIfaceCount=mediumIfaceCount,
        grids=UnsafePointer[Grid_C, MutAnyOrigin].unsafe_dangling(),
        gridCount=Int64(0),
        lightSampler=LightSampler_C(cdf=UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), n=Int32(0), _pad=Int32(0)),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, blasCount=blasCount,
        instances=instances, instanceCount=instanceCount,
        measuredBrdfs=measuredBrdfs, measuredBrdfCount=measuredBrdfCount,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        gpuTextures=gpuTextures, gpuTextureCount=gpuTextureCount,
    )

# ── Infinite/distant-light emission + NEE sampling (shared by bdpt.mojo and
#    sppm.mojo — lives here, not shading.mojo, to avoid an import cycle:
#    shading.mojo already imports helpers FROM sppm.mojo, so sppm.mojo can't
#    import back from shading.mojo. guide.mojo hit this exact same
#    constraint earlier and solved it by duplicating the two small equal-
#    area-mapping helpers below rather than sharing them — same approach
#    here.) ────────────────────────────────────────────────────────────────

@always_inline
def _scene_bounding_sphere(sd: SceneDescriptor2_C) -> Tuple[Point3f, Float32]:
    """Scene bounding sphere derived from the top-level BVH's root AABB.
    Infinite/distant lights have no position of their own — emitting a
    light-path/photon from one requires an arbitrary point outside the
    scene to start from; a disk of this radius (see
    _sample_disk_perpendicular) centered on the scene gives every emitted
    ray a chance to actually enter the scene, same technique pbrt uses
    (DistantLight::SampleLe / ImageInfiniteLight::SampleLe)."""
    var root = sd.bvh2Nodes[0]
    var diag = root.max - root.min
    var center = root.min + diag * Float32(0.5)
    var radius = diag.length() * Float32(0.5)
    if radius < Float32(1e-4):
        radius = Float32(1.0)
    return (center, radius)

@always_inline
def _sample_disk_perpendicular(
    dir: Vec3f,   # the ray's travel direction
    center: Point3f, radius: Float32,
    u: Point2f,
) -> Point3f:
    """Uniform point on a disk of `radius` centered at `center - dir*radius`
    (i.e. on the near side of the bounding sphere, facing back along `dir`)
    and oriented perpendicular to `dir` — so a ray started here travelling
    along `dir` is guaranteed to cross the whole bounding sphere."""
    var frame = Frame.from_z(dir)
    var r = radius * sqrt(u.x)
    var theta = Float32(2) * PI * u.y
    var dx = r * cos(theta)
    var dy = r * sin(theta)
    var disk_center = center + dir * (-radius)
    return disk_center + (frame.x * dx + frame.y * dy)

@always_inline
def _equal_area_square_to_sphere(u: Float32, v: Float32) -> SIMD[DType.float32, 3]:
    """PBRT v4's equal-area octahedral mapping (Clarberg 2008): [0,1]^2 UV ->
    unit sphere direction. THE single shared implementation — bvh.mojo is
    the only one of {shading.mojo, guide.mojo, bvh.mojo} with no import-
    cycle constraint against the other two (shading.mojo and guide.mojo
    both already import from bvh.mojo; bvh.mojo imports from neither), so
    this lives here and they import it, instead of each keeping its own
    copy. That prior 3-way duplication is exactly how a real bug survived
    undetected in 2 of the 3 copies after being found and fixed in the
    first: an extra `elif up >= vp: phi = (vp/r)*PI/4` branch that doesn't
    exist in pbrt-v4's own EqualAreaSquareToSphere (util/math.cpp) and
    computes a completely wrong angle for ~half of all directions —
    confirmed via a numerical round-trip test against
    _equal_area_sphere_to_square (max error 0.66 on the unit sphere before
    the fix, ~1e-6 after). This single formula (no up>=vp branch) matches
    pbrt exactly; r==0 fallback is PI/4, matching pbrt's `r==0 ? 1 : ...`
    ternary."""
    var uu = Float32(2) * u - Float32(1)
    var vv = Float32(2) * v - Float32(1)
    var up = uu if uu >= Float32(0) else -uu
    var vp = vv if vv >= Float32(0) else -vv
    var signed_dist = Float32(1) - (up + vp)
    var d = signed_dist if signed_dist >= Float32(0) else -signed_dist
    var r = Float32(1) - d
    var phi: Float32
    if r == Float32(0):
        phi = PI / Float32(4)
    else:
        phi = ((vp - up) / r + Float32(1)) * PI / Float32(4)
    var r2 = r * r
    var z = Float32(1) - r2
    if signed_dist < Float32(0): z = -z
    var cp = cos(phi)
    var sp = sin(phi)
    if uu < Float32(0): cp = -cp
    if vv < Float32(0): sp = -sp
    var xy_scale = r * sqrt(max(Float32(0), Float32(2) - r2))
    return SIMD[DType.float32, 3](cp * xy_scale, sp * xy_scale, z)

@always_inline
def _lower_bound_bvh(arr: UnsafePointer[Float32, MutAnyOrigin], lo: Int, hi: Int, val: Float32) -> Int:
    """Binary search: first index i in [lo, hi) s.t. arr[i] >= val; returns
    hi if all < val. Used only by _sample_infinite_light_textured below —
    the sole remaining copy after unifying the 3 duplicate CDF-walk sites
    (project_equal_area_mapping_bug memory, bug #4)."""
    var l = lo; var h = hi
    while l < h:
        var mid = (l + h) // 2
        if arr[mid] < val:
            l = mid + 1
        else:
            h = mid
    return l

@always_inline
def _sample_infinite_light_textured(
    ilight: InfiniteLight_C,
    u: Point2f,
) -> Tuple[Vec3f, RGB, Float32]:
    """CDF-importance-sample a TEXTURED environment light, returning
    (world-space direction, radiance there, solid-angle pdf). Callers must
    already know `ilight` has a valid texture+CDF (see the guard every call
    site checks: `tex_idx>=0 and pixels_ptr/cdf_ptr allocated and cdf_w>0`).

    THE single shared implementation of this CDF walk — previously
    duplicated byte-for-byte in 3 places (this function, shading.mojo's
    _nee_infinite_light, and shading.mojo's shade_coated_diffuse env-NEE
    block), which is exactly how an off-by-one bug in the CDF bucket lookup
    survived undetected: fixed in one copy without realizing 2 others
    existed. See project_equal_area_mapping_bug memory (bug #4) for the
    off-by-one's diagnosis — `_lower_bound`/`_lower_bound_bvh` return the
    first index i with cdf[i] >= val (the END of the bucket containing val),
    so the bucket index itself is i-1; every caller needs `- 1` (clamped to
    0) on both the row and column lookup."""
    var w2l = ilight.world_to_light
    var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
    var row_idx = _lower_bound_bvh(ilight.cdf_ptr, 0, ih, u.x) - 1
    row_idx = max(0, min(row_idx, ih - 1))
    var dp_row = ilight.cdf_ptr[row_idx + 1] - ilight.cdf_ptr[row_idx]
    var cond_base = (ih + 1) + row_idx * (iw + 1)
    var col_idx = _lower_bound_bvh(ilight.cdf_ptr, cond_base, cond_base + iw, u.y) - cond_base - 1
    col_idx = max(0, min(col_idx, iw - 1))
    var dp_col = ilight.cdf_ptr[cond_base + col_idx + 1] - ilight.cdf_ptr[cond_base + col_idx]

    var sample_u = (Float32(col_idx) + Float32(0.5)) / Float32(iw)
    var sample_v = (Float32(row_idx) + Float32(0.5)) / Float32(ih)
    var local_d_s = _equal_area_square_to_sphere(sample_u, sample_v)
    var local_d = Vec3f(local_d_s[0], local_d_s[1], local_d_s[2])

    var env_dir = Vec3f(
        w2l[0]*local_d.x + w2l[1]*local_d.y + w2l[2]*local_d.z,
        w2l[4]*local_d.x + w2l[5]*local_d.y + w2l[6]*local_d.z,
        w2l[8]*local_d.x + w2l[9]*local_d.y + w2l[10]*local_d.z,
    )

    var px = min(iw - 1, max(0, Int(sample_u * Float32(iw))))
    var py = min(ih - 1, max(0, Int(sample_v * Float32(ih))))
    var pr = ilight.pixels_ptr[(py*iw+px)*3+0]
    var pg = ilight.pixels_ptr[(py*iw+px)*3+1]
    var pb = ilight.pixels_ptr[(py*iw+px)*3+2]
    var env_rgb = RGB(pr, pg, pb) * ilight.scale

    var pdf_light: Float32
    if dp_row > Float32(0) and dp_col > Float32(0):
        pdf_light = dp_row * dp_col * Float32(iw * ih) / (Float32(4) * PI)
    else:
        pdf_light = Float32(1) / (Float32(4) * PI)
    return (env_dir, env_rgb, pdf_light)

@always_inline
def _sample_infinite_light_dir(
    ilight: InfiniteLight_C,
    u: Point2f,
) -> Tuple[Vec3f, RGB, Float32]:
    """Sample an emission direction from an environment (infinite) light,
    returning (world-space direction, radiance there, solid-angle pdf).
    Used for BOTH light-path/photon emission (bdpt.mojo/sppm.mojo) and NEE
    toward the light (same distribution works for both — the only
    difference is which end of the ray you start from)."""
    if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and _is_real_ptr(ilight.cdf_ptr) and ilight.cdf_w > Int32(0):
        return _sample_infinite_light_textured(ilight, u)
    else:
        # No texture: uniform sphere sampling, constant radiance.
        var cosT = Float32(2) * u.x - Float32(1)
        var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
        var phi = Float32(2) * PI * u.y
        var dir = Vec3f(sinT * cos(phi), sinT * sin(phi), cosT)
        return (dir, ilight.scale, Float32(1) / (Float32(4) * PI))

@fieldwise_init
struct LightSample(Copyable, Movable):
    """Uniform result of sampling ONE non-area light toward a shading point,
    for NEE — the "Light interface" this codebase's distant/point/sphere/
    infinite lights all present, so a single generic NEE-weight function
    (bxdf.mojo's _nee_weight_simple/_nee_weight_hair) can consume any of
    them. `valid=False` means "skip this sample" (degenerate distance,
    light behind the visible horizon, etc). `is_delta=True` (distant/point)
    means no competing BSDF-sampling strategy exists, so the caller applies
    MIS weight 1; `is_delta=False` (sphere/infinite) means the caller must
    weight against the material's own pdf via the power heuristic. Area
    lights are NOT covered here — their NEE is entangled with MNEE glass-
    refraction probing (see shading.mojo's _nee_area_lights) and stays a
    separate, diffuse-specific path."""
    var wi:       SIMD[DType.float32, 3]
    var Li:       RGB
    var pdf:      Float32
    var dist:     Float32
    var is_delta: Bool
    var valid:    Bool

@always_inline
def _invalid_light_sample() -> LightSample:
    return LightSample(SIMD[DType.float32, 3](0, 0, 0), RGB(0.0), Float32(1.0), Float32(0.0), True, False)

@always_inline
def _sample_distant_light_nee(dl: DistantLight_C) -> LightSample:
    """Distant (directional) light: delta position AND direction, same (wi,
    Li) everywhere in the scene. pdf=1 by convention (is_delta=True tells
    the caller to skip MIS entirely, so the value of pdf itself never
    matters beyond being nonzero)."""
    var ldir = SIMD[DType.float32, 3](dl.direction.x, dl.direction.y, dl.direction.z)
    var wi = -ldir
    return LightSample(wi, dl.emission, Float32(1.0), Float32(2000.0), True, True)

@always_inline
def _sample_point_light_nee(
    pl: PointLight_C,
    hit_point: SIMD[DType.float32, 3],
) -> LightSample:
    """Point light: delta position, 1/dist² falloff folded into Li so
    callers never need dist² separately."""
    var lpos = SIMD[DType.float32, 3](pl.position.x, pl.position.y, pl.position.z)
    var to_light = lpos - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    if dist <= Float32(0.0001):
        return _invalid_light_sample()
    var wi = to_light * (Float32(1.0) / dist)
    var Li = pl.intensity * (Float32(1.0) / dist_sq)
    return LightSample(wi, Li, Float32(1.0), dist, True, True)

@always_inline
def _sample_sphere_light_nee(
    sph: Sphere_C,
    n_sphere_lights: Int,
    hit_point: SIMD[DType.float32, 3],
    mut pcg: PCG32,
) -> LightSample:
    """Analytic sphere area light: uniform cone sampling over the visible
    solid angle (pbrt's Sphere::Sample(refPoint) technique). Ported verbatim
    from shading.mojo's/sppm.mojo's own (formerly duplicated 3x) sphere-
    light NEE block — a real (non-delta) pdf, caller must apply MIS."""
    if sph.isAreaLight == Int8(0):
        return _invalid_light_sample()
    var to_cx = sph.center.x - hit_point[0]
    var to_cy = sph.center.y - hit_point[1]
    var to_cz = sph.center.z - hit_point[2]
    var dc_sq = to_cx*to_cx + to_cy*to_cy + to_cz*to_cz
    var dc = sqrt(dc_sq)
    if dc < sph.radius or dc_sq == Float32(0.0):
        return _invalid_light_sample()
    var sin2_max = sph.radius * sph.radius / dc_sq
    if sin2_max >= Float32(1.0):
        return _invalid_light_sample()
    var cos_max = sqrt(Float32(1.0) - sin2_max)
    var zc = SIMD[DType.float32, 3](to_cx / dc, to_cy / dc, to_cz / dc)
    var xc: SIMD[DType.float32, 3]
    if (zc[0] if zc[0] >= Float32(0.0) else -zc[0]) < Float32(0.9):
        xc = cross(zc, SIMD[DType.float32, 3](Float32(1), Float32(0), Float32(0)))
    else:
        xc = cross(zc, SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0)))
    var xlen = dot(xc, xc)
    if xlen > Float32(0.0):
        xc = xc * (Float32(1.0) / sqrt(xlen))
    var yc = cross(zc, xc)
    var r1s = pcg.next_float()
    var r2s = pcg.next_float()
    var cos_th = Float32(1.0) - r1s * (Float32(1.0) - cos_max)
    var sin_th = sqrt(max(Float32(0.0), Float32(1.0) - cos_th * cos_th))
    var phis = TWO_PI * r2s
    var wi = xc * (sin_th * cos(phis)) + yc * (sin_th * sin(phis)) + zc * cos_th
    var wlen = dot(wi, wi)
    if wlen > Float32(0.0):
        wi = wi * (Float32(1.0) / sqrt(wlen))
    var solid_angle = TWO_PI * (Float32(1.0) - cos_max)
    var n_sph = Float32(max(n_sphere_lights, 1))
    var pdf_light = Float32(1.0) / (solid_angle * n_sph)
    return LightSample(wi, sph.emission, pdf_light, dc, False, True)

@always_inline
def _sample_infinite_light_nee(ilight: InfiniteLight_C, u: Point2f) -> LightSample:
    """Thin LightSample-shaped wrapper over the existing shared
    _sample_infinite_light_dir, so infinite lights present the same
    interface as the other 3 non-area light types above."""
    var (dir, Li, pdf) = _sample_infinite_light_dir(ilight, u)
    return LightSample(dir.to_simd(), Li, pdf, Float32(100000.0), False, True)

@always_inline
def _equal_area_sphere_to_square(dx: Float32, dy: Float32, dz: Float32) -> SIMD[DType.float32, 2]:
    """Inverse of _equal_area_square_to_sphere — see that function's
    docstring for why this is THE single shared implementation (bvh.mojo),
    imported by shading.mojo/guide.mojo rather than duplicated."""
    var x = dx if dx >= Float32(0) else -dx
    var y = dy if dy >= Float32(0) else -dy
    var z = dz if dz >= Float32(0) else -dz
    var r = sqrt(max(Float32(0), Float32(1) - z))
    var a = max(x, y)
    var b: Float32
    if a == Float32(0):
        b = Float32(0)
    else:
        b = min(x, y) / a
    var t1 = Float32(0.406758566246788489601959989e-5)
    var t2 = Float32(0.636226545274016134946890922156)
    var t3 = Float32(0.61572017898280213493197203466e-2)
    var t4 = Float32(-0.247333733281268944196501420480)
    var t5 = Float32(0.881770664775316294736387951347e-1)
    var t6 = Float32(0.419038818029165735901852432784e-1)
    var t7 = Float32(-0.251390972343483509333252996350e-1)
    var phi = t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))))
    if x < y:
        phi = Float32(1) - phi
    var v = phi * r
    var u = r - v
    if dz < Float32(0):
        var tmp = u
        u = Float32(1) - v
        v = Float32(1) - tmp
    if dx < Float32(0): u = -u
    if dy < Float32(0): v = -v
    u = Float32(0.5) * (u + Float32(1))
    v = Float32(0.5) * (v + Float32(1))
    return SIMD[DType.float32, 2](u, v)

@always_inline
def _eval_infinite_light_and_pdf(ilight: InfiniteLight_C, dir_world: Vec3f) -> Tuple[RGB, Float32]:
    """Radiance AND solid-angle sampling pdf an infinite (environment) light
    contributes along a ray travelling in `dir_world` — used for the
    camera-ray miss case (bdpt.mojo/sppm.mojo don't otherwise add any
    infinite-light contribution when a traced ray leaves the scene, unlike
    the ordinary unidirectional path tracer's own miss handler in
    shading.mojo, which this mirrors with a nearest-texel lookup instead of
    shading.mojo's bilinear one — consistent with _sample_infinite_light_dir's
    own nearest-texel sampling above, and a fine approximation for a single
    miss-ray sample). The pdf is needed to MIS-weight this escape strategy
    against NEE sampling the same light from the previous vertex — see
    bdpt.mojo's `last_bsdf_pdf` bookkeeping for why (getting this wrong
    double-counts unoccluded env light, the exact bug documented in
    [[project_infinite_light_shadows]])."""
    if ilight.tex_idx < Int32(0) or not _is_real_ptr(ilight.pixels_ptr) or ilight.cdf_w <= Int32(0):
        return (ilight.scale, Float32(1) / (Float32(4) * PI))
    var w2l = ilight.world_to_light
    var local_dir = Vec3f(
        w2l[0]*dir_world.x + w2l[4]*dir_world.y + w2l[8]*dir_world.z,
        w2l[1]*dir_world.x + w2l[5]*dir_world.y + w2l[9]*dir_world.z,
        w2l[2]*dir_world.x + w2l[6]*dir_world.y + w2l[10]*dir_world.z,
    )
    var uv = _equal_area_sphere_to_square(local_dir.x, local_dir.y, local_dir.z)
    var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
    var px = min(iw - 1, max(0, Int(uv[0] * Float32(iw))))
    var py = min(ih - 1, max(0, Int(uv[1] * Float32(ih))))
    var pr = ilight.pixels_ptr[(py*iw+px)*3+0]
    var pg = ilight.pixels_ptr[(py*iw+px)*3+1]
    var pb = ilight.pixels_ptr[(py*iw+px)*3+2]
    var radiance = RGB(pr, pg, pb) * ilight.scale
    var dp_row = ilight.cdf_ptr[py + 1] - ilight.cdf_ptr[py]
    var cond_base = (ih + 1) + py * (iw + 1)
    var dp_col = ilight.cdf_ptr[cond_base + px + 1] - ilight.cdf_ptr[cond_base + px]
    var pdf: Float32
    if dp_row > Float32(0) and dp_col > Float32(0):
        pdf = dp_row * dp_col * Float32(iw * ih) / (Float32(4) * PI)
    else:
        pdf = Float32(1) / (Float32(4) * PI)
    return (radiance, pdf)

# ── Marschner/Chiang 3-lobe hair BSDF — shared math ───────────────────────────
# Lives here (not shading.mojo) for the same import-cycle reason as
# _mk_sd_full above: shading.mojo already imports helpers FROM sppm.mojo, so
# sppm.mojo (and bdpt.mojo, which imports from sppm.mojo) can't import back
# from shading.mojo. shading.mojo's own shade_hair imports these from here
# instead of defining them locally — this is a pure relocation, no behavior
# change (same functions, same formulas).

@always_inline
def _hair_wrap(x: Float32) -> Float32:
    """Wrap angle to [-PI, PI] — GPU-safe, no unbounded loops."""
    return x - TWO_PI * floor((x + PI) / TWO_PI)

@always_inline
def _hair_asin(x: Float32) -> Float32:
    """Polynomial asin(x) for x in [-1,1] — avoids std.math asin which may not link on GPU.
    Uses sqrt-reflection for |x|>0.5 for accuracy; max error ~5e-6."""
    var ax = abs(x)
    var result: Float32
    if ax > Float32(0.5):
        var t = Float32(1.0) - ax
        var r = sqrt(Float32(2.0) * t)
        result = Float32(1.5707963) - r * (Float32(1.0) + t * (Float32(0.16666667) + t * (Float32(0.075) + t * Float32(0.04464286))))
    else:
        var x2 = ax * ax
        result = ax * (Float32(1.0) + x2 * (Float32(0.16666667) + x2 * (Float32(0.075) + x2 * Float32(0.04464286))))
    return -result if x < Float32(0.0) else result

@always_inline
def _hair_logistic(x: Float32, s: Float32) -> Float32:
    """Logistic distribution PDF centered at 0: exp(-|x|/s) / (s*(1+exp(-|x|/s))^2).
    Numerically stable via |x| form."""
    var e = exp(-abs(x) / s)
    return e / (s * (Float32(1.0) + e) * (Float32(1.0) + e))

@always_inline
def _hair_I0_poly(t: Float32) -> Float32:
    """9-term Taylor series for I0(x) where t = x*x/4. Accurate to 0.15% for t≤16 (x≤8)."""
    var t2 = t * t; var t3 = t2 * t; var t4 = t2 * t2; var t5 = t4 * t
    var t6 = t5 * t; var t7 = t6 * t; var t8 = t7 * t
    return Float32(1.0) + t + t2*Float32(0.25) + t3*Float32(0.027778) + t4*Float32(0.001736) + t5*Float32(6.944e-5) + t6*Float32(1.929e-6) + t7*Float32(3.930e-8) + t8*Float32(6.151e-10)

@always_inline
def _hair_Mp(cos_ti: Float32, cos_to: Float32, sin_ti: Float32, sin_to: Float32, inv_v: Float32, mp_c: Float32) -> Float32:
    """von Mises-Fisher Mp with precomputed constant mp_c = exp(-inv_v)/(2/inv_v) for v≤0.1,
    or 1/(sinh(inv_v)*2/inv_v) for v>0.1. inv_v = 1/v."""
    var a = cos_ti * cos_to * inv_v
    var b = sin_ti * sin_to * inv_v
    if a > Float32(8.0):
        # Asymptotic: I0(a) ≈ exp(a)/sqrt(2πa), so Mp ≈ mp_c * exp(a-b)/sqrt(2πa)
        return mp_c * exp(a - b) / sqrt(Float32(2.0) * PI * a)
    return mp_c * _hair_I0_poly(a * a * Float32(0.25)) * exp(-b)

@always_inline
def _atan2f(y: Float32, x: Float32) -> Float32:
    """atan2 via minimax polynomial — avoids the unresolved libdevice extern on GPU."""
    var ax = abs(x); var ay = abs(y)
    var mn = min(ax, ay)
    var mx = max(ax, ay)
    var a = mn / (mx if mx > Float32(1e-10) else Float32(1e-10))
    var s = a * a
    var r = (Float32(-0.0464964749) * s + Float32(0.15931422)) * s
    r = (r - Float32(0.327622764)) * s * a + a
    if ay > ax: r = Float32(1.5707963267948966) - r
    if x < Float32(0.0): r = Float32(3.14159265358979323846) - r
    if y < Float32(0.0): r = -r
    return r

@always_inline
def _hair_eval_lobes(
    wi: SIMD[DType.float32, 3],
    tangent: SIMD[DType.float32, 3],
    b_perp: SIMD[DType.float32, 3],
    n_perp: SIMD[DType.float32, 3],
    phi_o: Float32,
    dphi0: Float32, dphi1: Float32, dphi2: Float32,
    cos_tp0_o: Float32, sin_tp0_o: Float32,
    cos_tp1_o: Float32, sin_tp1_o: Float32,
    cos_tp2_o: Float32, sin_tp2_o: Float32,
    cos_theta_o: Float32, sin_theta_o: Float32,
    inv_vm0: Float32, inv_vm1: Float32, inv_vm2: Float32,
    mp_c0: Float32, mp_c1: Float32, mp_c2: Float32,
    s: Float32,
    A0: RGB, A1: RGB, A2: RGB, A3: RGB,
    lum0: Float32, lum1: Float32, lum2: Float32, lum3: Float32, total_lum: Float32,
) -> Tuple[Float32, RGB, Float32]:
    """Shared Marschner 3-lobe (R/TT/TRT) evaluation at an arbitrary direction
    wi. Returns (cos_ti, f, pdf_over_cos); pdf_over_cos is the lum-weighted
    M*N sum (already has the /cos_ti baked into M — multiply by cos_ti for
    the solid-angle PDF, as both NEE call sites and the indirect sampler do)."""
    var sin_ti = dot(wi, tangent)
    var cos_ti = max(safe_sqrt(Float32(1.0) - sin_ti * sin_ti), Float32(1e-5))
    var wi_perp = wi - sin_ti * tangent
    var phi_i = _atan2f(dot(wi_perp, b_perp), dot(wi_perp, n_perp))
    var dphi_i = phi_i - phi_o
    var x0 = _hair_wrap(dphi_i - dphi0)
    var x1 = _hair_wrap(dphi_i - dphi1)
    var x2 = _hair_wrap(dphi_i - dphi2)
    var M0 = _hair_Mp(cos_ti, cos_tp0_o, sin_ti, sin_tp0_o, inv_vm0, mp_c0) / cos_ti
    var M1 = _hair_Mp(cos_ti, cos_tp1_o, sin_ti, sin_tp1_o, inv_vm1, mp_c1) / cos_ti
    var M2 = _hair_Mp(cos_ti, cos_tp2_o, sin_ti, sin_tp2_o, inv_vm2, mp_c2) / cos_ti
    var M3 = _hair_Mp(cos_ti, cos_theta_o, sin_ti, sin_theta_o, inv_vm2, mp_c2) / cos_ti
    var N0 = _hair_logistic(x0, s)
    var N1 = _hair_logistic(x1, s)
    var N2 = _hair_logistic(x2, s)
    var N3 = Float32(0.5) * INV_PI
    var f = A0*(M0*N0) + A1*(M1*N1) + A2*(M2*N2) + A3*(M3*N3)
    var pdf_over_cos = (lum0*M0*N0 + lum1*M1*N1 + lum2*M2*N2 + lum3*M3*N3) / total_lum
    return (cos_ti, f, pdf_over_cos)

@fieldwise_init
struct HairLobeConstants(TrivialRegisterPassable):
    """Precomputed per-hit Marschner/Chiang 3-lobe hair BSDF constants —
    everything _hair_eval_lobes/_hair_sample_dir need that depends only on
    the hit geometry + wo + material, NOT on wi. Recomputed fresh at each
    connect/gather/NEE call site (via _hair_precompute) from a handful of
    persisted scalars (curve index, h, v, wo, material index) rather than
    stored inline in BDPTVertex/SPPMPixel — those structs stay small, and
    this recompute is the same cost class as bdpt.mojo's own
    _eval_conductor_ggx per-call GGX evaluation."""
    var tangent: SIMD[DType.float32, 3]
    var b_perp:  SIMD[DType.float32, 3]
    var n_perp:  SIMD[DType.float32, 3]
    var geo_normal: SIMD[DType.float32, 3]
    var phi_o: Float32
    var dphi0: Float32
    var dphi1: Float32
    var dphi2: Float32
    var cos_tp0_o: Float32
    var sin_tp0_o: Float32
    var cos_tp1_o: Float32
    var sin_tp1_o: Float32
    var cos_tp2_o: Float32
    var sin_tp2_o: Float32
    var cos_theta_o: Float32
    var sin_theta_o: Float32
    var vm0: Float32
    var vm1: Float32
    var vm2: Float32
    var inv_vm0: Float32
    var inv_vm1: Float32
    var inv_vm2: Float32
    var mp_c0: Float32
    var mp_c1: Float32
    var mp_c2: Float32
    var e2vm0: Float32
    var e2vm1: Float32
    var e2vm2: Float32
    var s: Float32
    var A0: RGB
    var A1: RGB
    var A2: RGB
    var A3: RGB
    var lum0: Float32
    var lum1: Float32
    var lum2: Float32
    var lum3: Float32
    var total_lum: Float32
    var radius: Float32   # local curve radius at the hit piece; scales self-intersection offset epsilon

@always_inline
def _hair_precompute(
    mat: Material_C,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curve_idx: Int,
    v_global: Float32,
    h_raw: Float32,
    wo: SIMD[DType.float32, 3],
) -> HairLobeConstants:
    """Recomputes shading.mojo's shade_hair Steps 1-10 (fiber frame, optical
    quantities, per-lobe attenuation/variance/azimuthal-peak precompute) from
    just the curve hit + wo + material. Material packing (matches shade_hair):
    albedo = sigma_a (RGB absorption), emission.r = eta (IOR), roughU = betaM
    (longitudinal), roughV = betaN (azimuthal). Shared verbatim by
    shading.mojo's own shade_hair (the primary path tracer) and
    bdpt.mojo/sppm.mojo's connectible-vertex/gather/NEE evaluation of a
    stored hair vertex."""
    var h = max(Float32(-0.99), min(Float32(0.99), h_raw))
    var curve = curves[curve_idx]
    var piece = min(Int(curve.n_pieces) - 1, max(0, Int(v_global * Float32(curve.n_pieces))))
    var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
    var radius = (r0 + r1) * Float32(0.5)
    var seg_axis = q1 - q0
    var seg_len = sqrt(dot(seg_axis, seg_axis))
    var tangent: SIMD[DType.float32, 3]
    if seg_len > Float32(1e-8):
        tangent = seg_axis * (Float32(1.0) / seg_len)
    else:
        tangent = SIMD[DType.float32, 3](Float32(1), Float32(0), Float32(0))
    var n_perp = _curve_perp_axis(tangent)
    var b_perp = cross(tangent, n_perp)
    var geo_normal = n_perp * h + b_perp * sqrt(max(Float32(0.0), Float32(1.0) - h*h))

    var sin_theta_o = dot(wo, tangent)
    var cos_theta_o = safe_sqrt(Float32(1.0) - sin_theta_o * sin_theta_o)
    cos_theta_o = max(cos_theta_o, Float32(1e-5))

    var wo_perp = wo - sin_theta_o * tangent
    var phi_o = _atan2f(dot(wo_perp, b_perp), dot(wo_perp, n_perp))

    var eta = mat.emission.r
    var sigma_a = mat.albedo
    var betaM = mat.roughU
    var betaN = mat.roughV

    var h_clamped = max(Float32(-1.0) + Float32(1e-5), min(Float32(1.0) - Float32(1e-5), h))
    var gamma_o = _hair_asin(h_clamped)

    var eta_sq = eta * eta
    var sin_to_sq = sin_theta_o * sin_theta_o
    var eta_perp_num = max(Float32(0.0), eta_sq - sin_to_sq)
    var eta_perp = safe_sqrt(eta_perp_num) / cos_theta_o

    var sin_gamma_t_raw = sin(gamma_o) / max(eta_perp, Float32(1.0))
    var sin_gamma_t = max(Float32(-1.0) + Float32(1e-5), min(Float32(1.0) - Float32(1e-5), sin_gamma_t_raw))
    var gamma_t = _hair_asin(sin_gamma_t)
    var cos_gamma_t = safe_sqrt(Float32(1.0) - sin_gamma_t * sin_gamma_t)

    var T = RGB(
        exp(-sigma_a.r * Float32(2.0) * cos_gamma_t),
        exp(-sigma_a.g * Float32(2.0) * cos_gamma_t),
        exp(-sigma_a.b * Float32(2.0) * cos_gamma_t),
    )

    var cos_gamma_o = safe_sqrt(Float32(1.0) - h_clamped * h_clamped)
    var fr = fr_dielectric(cos_theta_o * cos_gamma_o, eta)
    var omfr = Float32(1.0) - fr

    var A0 = RGB(fr, fr, fr)
    var A1 = T * (omfr * omfr)
    var A2 = T * T * (omfr * omfr * fr)
    var A3 = RGB(
        A2.r * fr * T.r / max(Float32(1e-6), Float32(1.0) - fr * T.r),
        A2.g * fr * T.g / max(Float32(1e-6), Float32(1.0) - fr * T.g),
        A2.b * fr * T.b / max(Float32(1e-6), Float32(1.0) - fr * T.b),
    )

    var sin2k_0 = Float32(0.034899)
    var cos2k_0 = Float32(0.999391)
    var sin2k_1 = Float32(2.0) * cos2k_0 * sin2k_0
    var cos2k_1 = cos2k_0 * cos2k_0 - sin2k_0 * sin2k_0
    var sin2k_2 = Float32(2.0) * cos2k_1 * sin2k_1
    var cos2k_2 = cos2k_1 * cos2k_1 - sin2k_1 * sin2k_1
    var sin_tp0_o = sin_theta_o * cos2k_1 - cos_theta_o * sin2k_1
    var cos_tp0_o = cos_theta_o * cos2k_1 + sin_theta_o * sin2k_1
    var sin_tp1_o = sin_theta_o * cos2k_0 + cos_theta_o * sin2k_0
    var cos_tp1_o = cos_theta_o * cos2k_0 - sin_theta_o * sin2k_0
    var sin_tp2_o = sin_theta_o * cos2k_2 + cos_theta_o * sin2k_2
    var cos_tp2_o = cos_theta_o * cos2k_2 - sin_theta_o * sin2k_2

    var bm2 = betaM * betaM
    var bm4 = bm2 * bm2; var bm8 = bm4 * bm4; var bm16 = bm8 * bm8
    var bm20 = bm16 * bm4
    var tmp_v = Float32(0.726) * betaM + Float32(0.812) * bm2 + Float32(3.7) * bm20
    var vm0 = max(tmp_v * tmp_v, Float32(0.0001))
    var vm1 = max(vm0 * Float32(0.25), Float32(0.0001))
    var vm2 = max(vm0 * Float32(4.0), Float32(0.0001))
    var inv_vm0 = Float32(1.0) / vm0; var inv_vm1 = Float32(1.0) / vm1; var inv_vm2 = Float32(1.0) / vm2
    var mp_c0 = exp(-inv_vm0) / (Float32(2.0) * vm0)
    var mp_c1 = exp(-inv_vm1) / (Float32(2.0) * vm1)
    var sinh_inv2 = (exp(inv_vm2) - exp(-inv_vm2)) * Float32(0.5)
    var mp_c2 = Float32(1.0) / (sinh_inv2 * Float32(2.0) * vm2)
    var e2vm0 = exp(-Float32(2.0) * inv_vm0)
    var e2vm1 = exp(-Float32(2.0) * inv_vm1)
    var e2vm2 = exp(-Float32(2.0) * inv_vm2)

    var dphi0 = -Float32(2.0) * gamma_o
    var dphi1 = -PI + Float32(2.0) * (gamma_t - gamma_o)
    var dphi2 = Float32(4.0) * gamma_t - Float32(2.0) * gamma_o

    var s = betaN

    var lum0 = A0.luma() + Float32(1e-6)
    var lum1 = A1.luma() + Float32(1e-6)
    var lum2 = A2.luma() + Float32(1e-6)
    var lum3 = A3.luma() + Float32(1e-6)
    var total_lum = lum0 + lum1 + lum2 + lum3

    return HairLobeConstants(
        tangent=tangent, b_perp=b_perp, n_perp=n_perp, geo_normal=geo_normal,
        radius=radius,
        phi_o=phi_o, dphi0=dphi0, dphi1=dphi1, dphi2=dphi2,
        cos_tp0_o=cos_tp0_o, sin_tp0_o=sin_tp0_o,
        cos_tp1_o=cos_tp1_o, sin_tp1_o=sin_tp1_o,
        cos_tp2_o=cos_tp2_o, sin_tp2_o=sin_tp2_o,
        cos_theta_o=cos_theta_o, sin_theta_o=sin_theta_o,
        vm0=vm0, vm1=vm1, vm2=vm2,
        inv_vm0=inv_vm0, inv_vm1=inv_vm1, inv_vm2=inv_vm2,
        mp_c0=mp_c0, mp_c1=mp_c1, mp_c2=mp_c2,
        e2vm0=e2vm0, e2vm1=e2vm1, e2vm2=e2vm2,
        s=s, A0=A0, A1=A1, A2=A2, A3=A3,
        lum0=lum0, lum1=lum1, lum2=lum2, lum3=lum3, total_lum=total_lum,
    )

@always_inline
def curve_offset_eps(radius: Float32) -> Float32:
    """Self-intersection offset for hair/curve shadow-ray origins and bounce
    continuations, scaled to the local fiber radius rather than a fixed
    universal constant. A flat `0.0001` offset was wrong in both directions
    depending on the scene's fiber radius (too small relative to thick
    strands to reliably clear the fiber's own curved surface; too large
    relative to fine strands, jumping clear outside a fiber's silhouette).
    A full local radius overshoots the other way in densely-packed fur/hair
    (bunny-fur, curly-hair): adjacent strands are typically only a few
    radii apart, so offsetting by a whole radius routinely lands the ray
    origin inside a *neighboring* strand instead of just clearing the
    current one, causing systematic over-occlusion (verified empirically —
    1.0x turned the "hair" scene solid black and "bunny-fur" bald/smooth).
    A small fraction of the radius is enough to escape the current fiber's
    own surface (the self-intersection problem this exists to solve) without
    reaching far enough to routinely hit a neighbor; the floor guards the
    degenerate near-zero-radius case."""
    return max(radius, Float32(1e-6))

@always_inline
def _hair_sample_dir(
    hc: HairLobeConstants,
    mut pcg: PCG32,
) -> Tuple[SIMD[DType.float32, 3], RGB, Float32, Float32]:
    """Importance-samples a continuation direction from the 3-lobe hair BSDF
    (lobe picked proportional to luminance, vMF longitudinal + logistic
    azimuthal sampling) — extracted from shading.mojo's shade_hair Step 15
    verbatim, same RNG draw order (r_lobe, u_th, u_phi_v, u3), so it's a
    drop-in replacement there and reusable by bdpt.mojo/sppm.mojo for
    continuing a path through a stored hair vertex (mirrors
    bxdf_sample_conductor's role for conductor). Returns
    (wi, f, pdf_over_cos, cos_ti); pdf_over_cos already has the /cos_ti
    baked in (see _hair_eval_lobes), so the correct importance-sampling
    throughput update is `beta *= f / pdf_over_cos` — cos_ti cancels, same
    as shade_hair's own `throughput *= f * (1/pdf)`."""
    var r_lobe = pcg.next_float() * hc.total_lum
    var vm_p: Float32
    var e2vm_p: Float32
    var dphi_p: Float32
    var sin_tp_o_s: Float32
    var cos_tp_o_s: Float32
    var uniform_phi_s = False
    if r_lobe < hc.lum0:
        vm_p = hc.vm0; e2vm_p = hc.e2vm0; dphi_p = hc.dphi0
        sin_tp_o_s = hc.sin_tp0_o; cos_tp_o_s = hc.cos_tp0_o
    elif r_lobe < hc.lum0 + hc.lum1:
        vm_p = hc.vm1; e2vm_p = hc.e2vm1; dphi_p = hc.dphi1
        sin_tp_o_s = hc.sin_tp1_o; cos_tp_o_s = hc.cos_tp1_o
    elif r_lobe < hc.lum0 + hc.lum1 + hc.lum2:
        vm_p = hc.vm2; e2vm_p = hc.e2vm2; dphi_p = hc.dphi2
        sin_tp_o_s = hc.sin_tp2_o; cos_tp_o_s = hc.cos_tp2_o
    else:
        vm_p = hc.vm2; e2vm_p = hc.e2vm2; dphi_p = Float32(0.0)
        sin_tp_o_s = hc.sin_theta_o; cos_tp_o_s = hc.cos_theta_o
        uniform_phi_s = True

    var u_th = max(pcg.next_float(), Float32(1e-6))
    var u_phi_v = pcg.next_float()
    var cos_th = Float32(1.0) + vm_p * log(u_th + (Float32(1.0) - u_th) * e2vm_p)
    cos_th = max(Float32(-1.0), min(Float32(1.0), cos_th))
    var sin_th = safe_sqrt(Float32(1.0) - cos_th * cos_th)
    var cos_phi_v = cos(TWO_PI * u_phi_v)
    var sin_ti_s = max(Float32(-1.0), min(Float32(1.0), -cos_th * sin_tp_o_s + sin_th * cos_phi_v * cos_tp_o_s))
    var cos_ti_s = safe_sqrt(Float32(1.0) - sin_ti_s * sin_ti_s)
    cos_ti_s = max(cos_ti_s, Float32(1e-5))

    var u3 = max(pcg.next_float(), Float32(1e-6))
    var phi_i_s: Float32
    if uniform_phi_s:
        phi_i_s = hc.phi_o + TWO_PI * u3 - PI
    else:
        phi_i_s = hc.phi_o + dphi_p + hc.s * log(u3 / max(Float32(1.0) - u3, Float32(1e-6)))

    var wi_s = sin_ti_s * hc.tangent + cos(phi_i_s) * cos_ti_s * hc.n_perp + sin(phi_i_s) * cos_ti_s * hc.b_perp
    var wi_len = dot(wi_s, wi_s)
    if wi_len > Float32(0.0):
        wi_s = wi_s * (Float32(1.0) / sqrt(wi_len))

    var (cos_ti_s2, f_s, pdf_over_cos_s) = _hair_eval_lobes(
        wi_s, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
        hc.dphi0, hc.dphi1, hc.dphi2,
        hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
        hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
        hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
    )
    var pdf = max(pdf_over_cos_s, Float32(1e-6))
    return (wi_s, f_s, pdf, cos_ti_s2)

# ── Analytical sphere intersection ────────────────────────────────────────────

@always_inline
def ray_sphere_hit(center: Point3f, radius: Float32,
                  ray: Ray_C, t_min: Float32, t_max: Float32) -> Float32:
    """Exact ray-sphere intersection. Returns t of first hit in (t_min, t_max), or -1."""
    var ocx = ray.origin.x - center.x
    var ocy = ray.origin.y - center.y
    var ocz = ray.origin.z - center.z
    # Use half-b form to avoid catastrophic cancellation
    var a = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y + ray.direction.z*ray.direction.z
    var half_b = ocx*ray.direction.x + ocy*ray.direction.y + ocz*ray.direction.z
    var c = ocx*ocx + ocy*ocy + ocz*ocz - radius * radius
    var disc = half_b * half_b - a * c
    if disc < Float32(0.0):
        return Float32(-1.0)
    var sqrtd = sqrt(disc)
    var t = (-half_b - sqrtd) / a
    if t >= t_min and t < t_max:
        return t
    t = (-half_b + sqrtd) / a
    if t >= t_min and t < t_max:
        return t
    return Float32(-1.0)

@always_inline
def test_spheres(
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    ray: Ray_C,
    result: UnsafePointer[Intersection_C, MutAnyOrigin],
):
    """Test all analytical spheres against the ray, updating result if closer.
    Sets primId.type = 4 and primId.id1 = sphere_index on a sphere hit.
    """
    var t_max = Float32(1.0e38)
    if result[0].hit != Int8(0):
        t_max = result[0].tHit
    for i in range(n_spheres):
        var t = ray_sphere_hit(spheres[i].center, spheres[i].radius, ray, Float32(1e-4), t_max)
        if t > Float32(0.0):
            t_max = t
            result[0].hit = Int8(1)
            result[0].tHit = t
            result[0].primId.type = Int8(4)
            result[0].primId.id1 = Int64(i)
            result[0].primId.materialIndex = Int64(spheres[i].materialIndex)
            result[0].primId.instanceIdx = Int32(-1)  # spheres are never instanced; clear any stale value

# ── Object-instance BLAS walk ──────────────────────────────────────────────
# A BLAS only ever contains ordinary type==0 triangles (built by finalize_scene
# from one template's mesh range), so this is a plain single-level walk — never
# recurses into another BLAS or handles spheres/curves/instances. Called from
# the type==6 branch below with a ray already transformed into the instance's
# object space (direction NOT renormalized, so tHit stays comparable to the
# caller's world-space ray parameterization).

@always_inline
def _traverse_blas_triangles(
    blasNodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    blasPrimIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    tMax: Float32,
) -> Tuple[Bool, Float32, Float32, Float32, PrimId_C]:
    """Returns (hit, tHit, u, v, primId) — primId is the winning BLAS-local
    type==0 entry as-is (mesh_idx/base_vidx/materialIndex already correct
    against the shared global `meshes` array); the caller overwrites its
    `instanceIdx` field before use."""
    var rdir = Vec3f(Float32(1.0) / ray_dir[0], Float32(1.0) / ray_dir[1], Float32(1.0) / ray_dir[2])
    var org = Vec3f(ray_org[0], ray_org[1], ray_org[2])
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)

    var hasHit = False
    var hitPrim = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0

    var stack = InlineArray[Int32, 64](fill=Int32(0))
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0

    while True:
        var node = blasNodes[current]
        if node.count > 0:
            var offset = Int(node.offset)
            var count = Int(node.count)
            for j in range(count):
                var prim = blasPrimIds[offset + j]
                if prim.type != Int8(0):
                    continue
                var mesh_idx = Int(prim.id1)
                var base_vidx = Int(prim.id2)
                var mesh = meshes[mesh_idx]
                var v0 = Int(mesh.vertexIndices[base_vidx])
                var v1 = Int(mesh.vertexIndices[base_vidx + 1])
                var v2 = Int(mesh.vertexIndices[base_vidx + 2])
                var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
                var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
                var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
                var hit_res = intersect_triangle(ray_org, ray_dir, p0, p1, p2, localTHit)
                if hit_res[0]:
                    localTHit = hit_res[1]
                    bestU = hit_res[2]
                    bestV = hit_res[3]
                    hitPrim = prim
                    hasHit = True
            if toVisit == 0:
                break
            toVisit -= 1
            current = Int(stack_ptr[toVisit])
        else:
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)
            var leftNode = blasNodes[leftIdx]
            var rightNode = blasNodes[rightIdx]
            var leftHit = intersect_aabb(
                leftNode.min, leftNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]
            if leftIsHit and rightIsHit:
                if leftHit[1] <= rightHit[1]:
                    current = leftIdx
                    stack_ptr[toVisit] = Int32(rightIdx)
                else:
                    current = rightIdx
                    stack_ptr[toVisit] = Int32(leftIdx)
                toVisit += 1
            elif leftIsHit:
                current = leftIdx
            elif rightIsHit:
                current = rightIdx
            else:
                if toVisit == 0:
                    break
                toVisit -= 1
                current = Int(stack_ptr[toVisit])

    return (hasHit, localTHit, bestU, bestV, hitPrim)


@always_inline
def _traverse_instance_leaf(
    prim: PrimId_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    tMax: Float32,
) -> Tuple[Bool, Float32, Float32, Float32, PrimId_C]:
    """Handle one PrimId_C.type==6 (instance) leaf: transform the ray into the
    instance's object space and walk its BLAS. Returns (hit, tHit, u, v,
    primId) with primId.instanceIdx already set to this instance's index.
    Shared by all three top-level traversal functions (traverse_bvh2_core,
    any_hit_bvh2_core, traverse_bvh2_core_defer_curves) so a correctness fix
    here only needs to happen once. Returns hit=False without dereferencing
    anything if instance data isn't available (callers that never populate it
    — e.g. GPU's own device-side scene upload, which reads the instance-free
    TLAS instead — pass dangling defaults; see [[project_object_instancing]]
    for why an unguarded dereference here previously crashed the GPU path)."""
    var dummy = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    if Int(instances) <= 4 or Int(blasNodesArr) <= 4 or Int(blasPrimIdsArr) <= 4:
        return (False, tMax, Float32(0), Float32(0), dummy)
    var inst_idx = Int(prim.id1)
    var inst = instances[inst_idx]
    var (o_org, o_dir) = _transform_ray_to_instance_space(inst.worldToObj, ray_org, ray_dir)
    var blas_nodes = blasNodesArr[Int(inst.blasIdx)]
    var blas_prim_ids = blasPrimIdsArr[Int(inst.blasIdx)]
    var sub = _traverse_blas_triangles(blas_nodes, blas_prim_ids, meshes, o_org, o_dir, tMax)
    if sub[0]:
        var hit_prim = sub[4]
        hit_prim.instanceIdx = Int32(inst_idx)
        return (True, sub[1], sub[2], sub[3], hit_prim)
    return (False, tMax, Float32(0), Float32(0), dummy)


@always_inline
def _transform_ray_to_instance_space(
    worldToObj: SIMD[DType.float32, 16],
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
) -> Tuple[SIMD[DType.float32, 3], SIMD[DType.float32, 3]]:
    """Transform a world-space ray into an instance's object space. The
    direction is rotated/scaled but NOT renormalized and the origin is NOT
    re-based on it — this preserves the ray parameterization so a `tHit` found
    in object space is directly usable as the world-space `tHit` (same trick
    pbrt's TransformedPrimitive uses)."""
    var m = worldToObj
    var ox = m[0]*ray_org[0] + m[4]*ray_org[1] + m[8]*ray_org[2]  + m[12]
    var oy = m[1]*ray_org[0] + m[5]*ray_org[1] + m[9]*ray_org[2]  + m[13]
    var oz = m[2]*ray_org[0] + m[6]*ray_org[1] + m[10]*ray_org[2] + m[14]
    var dx = m[0]*ray_dir[0] + m[4]*ray_dir[1] + m[8]*ray_dir[2]
    var dy = m[1]*ray_dir[0] + m[5]*ray_dir[1] + m[9]*ray_dir[2]
    var dz = m[2]*ray_dir[0] + m[6]*ray_dir[1] + m[10]*ray_dir[2]
    return (SIMD[DType.float32, 3](ox, oy, oz), SIMD[DType.float32, 3](dx, dy, dz))


# ── Unified traversal core (CPU + GPU) ────────────────────────────────────────

@always_inline
def traverse_bvh2_core(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
    resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
):
    # Callers that never populate any PrimId_C.type==6 leaf (GPU kernels, which
    # upload their own device-side scene copy with no instance data at all) can
    # omit blasNodesArr/blasPrimIdsArr/instances entirely — the dangling
    # defaults above are never dereferenced since no such leaf will exist.

    var rdir = Vec3f(Float32(1.0) / ray.direction.x, Float32(1.0) / ray.direction.y, Float32(1.0) / ray.direction.z)
    var org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)

    var hitIndex: Int = -1
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0
    var instHit = False
    var instHitPrim = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))

    var stack = InlineArray[Int32, 64](fill=Int32(0))
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0

    var ray_org = SIMD[DType.float32, 3](ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = SIMD[DType.float32, 3](ray.direction.x, ray.direction.y, ray.direction.z)

    while True:
        var node = bvh2Nodes[current]

        if node.count > 0:
            # Leaf node — intersect primitives
            var offset = Int(node.offset)
            var count = Int(node.count)
            for j in range(count):
                var prim = primIds[offset + j]
                var mesh_idx: Int
                var base_vidx: Int

                if prim.type == 0:
                    mesh_idx = Int(prim.id1)
                    base_vidx = Int(prim.id2)
                elif prim.type == 1 or prim.type == 2 or prim.type == 3:
                    if prim.id2 == -1:
                        continue # GPU cannot intersect non-triangle shapes directly yet
                    mesh_idx = Int(prim.id2 >> 32)
                    base_vidx = Int(prim.id2 & 0xFFFFFFFF) * 3
                elif prim.type == 5:
                    var curve = curves[Int(prim.id1)]
                    var curve_hit = intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, localTHit)
                    if curve_hit[0]:
                        localTHit = curve_hit[1]
                        bestU = curve_hit[2]
                        bestV = curve_hit[3]
                        hitIndex = offset + j
                        instHit = False
                    continue
                elif prim.type == 6:
                    var sub = _traverse_instance_leaf(prim, meshes, blasNodesArr, blasPrimIdsArr, instances, ray_org, ray_dir, localTHit)
                    if sub[0]:
                        localTHit = sub[1]
                        bestU = sub[2]
                        bestV = sub[3]
                        instHitPrim = sub[4]
                        instHit = True
                    continue
                else:
                    continue

                var mesh = meshes[mesh_idx]
                var v0_idx = Int(mesh.vertexIndices[base_vidx])
                var v1_idx = Int(mesh.vertexIndices[base_vidx + 1])
                var v2_idx = Int(mesh.vertexIndices[base_vidx + 2])

                var p0 = SIMD[DType.float32, 3](
                    mesh.points[v0_idx * 4],
                    mesh.points[v0_idx * 4 + 1],
                    mesh.points[v0_idx * 4 + 2]
                )
                var p1 = SIMD[DType.float32, 3](
                    mesh.points[v1_idx * 4],
                    mesh.points[v1_idx * 4 + 1],
                    mesh.points[v1_idx * 4 + 2]
                )
                var p2 = SIMD[DType.float32, 3](
                    mesh.points[v2_idx * 4],
                    mesh.points[v2_idx * 4 + 1],
                    mesh.points[v2_idx * 4 + 2]
                )

                var hit_res = intersect_triangle(ray_org, ray_dir, p0, p1, p2, localTHit)
                if hit_res[0]:
                    localTHit = hit_res[1]
                    bestU = hit_res[2]
                    bestV = hit_res[3]
                    hitIndex = offset + j
                    instHit = False

            # Pop next node from stack
            if toVisit == 0:
                break
            toVisit -= 1
            current = Int(stack_ptr[toVisit])
        else:
            # Interior node — test both children, visit nearer first
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)

            var leftNode = bvh2Nodes[leftIdx]
            var rightNode = bvh2Nodes[rightIdx]

            var leftHit = intersect_aabb(
                leftNode.min, leftNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )

            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]

            if leftIsHit and rightIsHit:
                # Both hit — visit nearer first, push farther
                var leftTNear = leftHit[1]
                var rightTNear = rightHit[1]
                if leftTNear <= rightTNear:
                    current = leftIdx
                    stack_ptr[toVisit] = Int32(rightIdx)
                else:
                    current = rightIdx
                    stack_ptr[toVisit] = Int32(leftIdx)
                toVisit += 1
            elif leftIsHit:
                current = leftIdx
            elif rightIsHit:
                current = rightIdx
            else:
                # Neither child hit — pop from stack
                if toVisit == 0:
                    break
                toVisit -= 1
                current = Int(stack_ptr[toVisit])

    if instHit:
        resultPtr[0] = Intersection_C(instHitPrim, localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    elif hitIndex != -1:
        resultPtr[0] = Intersection_C(primIds[hitIndex], localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    else:
        var dummyId = PrimId_C(-1, -1, 0, -1, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)


# Divergence-mitigation experiment for traverse_paths_gpu: identical traversal to
# traverse_bvh2_core, except curve leaves are not intersected inline. Instead, up
# to CURVE_DEFER_K candidate prim indices are recorded per ray (curve_cand_prim/
# curve_cand_count, both already offset to this ray's slot by the caller) so the
# expensive ray-vs-cylinder math can run later in a compacted kernel over just the
# rays that actually touched curve geometry, instead of scattered across every
# warp in the main traversal kernel. ncu profiling showed traverse_paths_gpu on
# fur scenes averaging 2.63/32 active threads per warp and 73% uncoalesced global
# sectors — both are classic per-ray-scalar-divergent-leaf-work symptoms, not
# register pressure (three register/inlining fixes measured no improvement).
# On overflow (>CURVE_DEFER_K candidates for one ray) this falls back to the
# immediate intersect_curve call, so correctness never depends on K being large
# enough — it only affects how many rays get the compaction benefit.
@always_inline
def traverse_bvh2_core_defer_curves(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
    resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin],
    curve_cand_prim: UnsafePointer[Int32, MutAnyOrigin],
    curve_cand_count: UnsafePointer[Int32, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
):

    var rdir = Vec3f(Float32(1.0) / ray.direction.x, Float32(1.0) / ray.direction.y, Float32(1.0) / ray.direction.z)
    var org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)

    var hitIndex: Int = -1
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0
    var instHit = False
    var instHitPrim = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))

    var stack = InlineArray[Int32, 64](fill=Int32(0))
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0

    var ray_org = SIMD[DType.float32, 3](ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = SIMD[DType.float32, 3](ray.direction.x, ray.direction.y, ray.direction.z)

    while True:
        var node = bvh2Nodes[current]

        if node.count > 0:
            # Leaf node — intersect primitives
            var offset = Int(node.offset)
            var count = Int(node.count)
            for j in range(count):
                var prim = primIds[offset + j]
                var mesh_idx: Int
                var base_vidx: Int

                if prim.type == 0:
                    mesh_idx = Int(prim.id1)
                    base_vidx = Int(prim.id2)
                elif prim.type == 1 or prim.type == 2 or prim.type == 3:
                    if prim.id2 == -1:
                        continue # GPU cannot intersect non-triangle shapes directly yet
                    mesh_idx = Int(prim.id2 >> 32)
                    base_vidx = Int(prim.id2 & 0xFFFFFFFF) * 3
                elif prim.type == 5:
                    var slot = curve_cand_count[0]
                    if slot < Int32(CURVE_DEFER_K):
                        curve_cand_prim[Int(slot)] = Int32(offset + j)
                        curve_cand_count[0] = slot + Int32(1)
                    else:
                        var curve = curves[Int(prim.id1)]
                        var curve_hit = intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, localTHit)
                        if curve_hit[0]:
                            localTHit = curve_hit[1]
                            bestU = curve_hit[2]
                            bestV = curve_hit[3]
                            hitIndex = offset + j
                            instHit = False
                    continue
                elif prim.type == 6:
                    var sub = _traverse_instance_leaf(prim, meshes, blasNodesArr, blasPrimIdsArr, instances, ray_org, ray_dir, localTHit)
                    if sub[0]:
                        localTHit = sub[1]
                        bestU = sub[2]
                        bestV = sub[3]
                        instHitPrim = sub[4]
                        instHit = True
                    continue
                else:
                    continue

                var mesh = meshes[mesh_idx]
                var v0_idx = Int(mesh.vertexIndices[base_vidx])
                var v1_idx = Int(mesh.vertexIndices[base_vidx + 1])
                var v2_idx = Int(mesh.vertexIndices[base_vidx + 2])

                var p0 = SIMD[DType.float32, 3](
                    mesh.points[v0_idx * 4],
                    mesh.points[v0_idx * 4 + 1],
                    mesh.points[v0_idx * 4 + 2]
                )
                var p1 = SIMD[DType.float32, 3](
                    mesh.points[v1_idx * 4],
                    mesh.points[v1_idx * 4 + 1],
                    mesh.points[v1_idx * 4 + 2]
                )
                var p2 = SIMD[DType.float32, 3](
                    mesh.points[v2_idx * 4],
                    mesh.points[v2_idx * 4 + 1],
                    mesh.points[v2_idx * 4 + 2]
                )

                var hit_res = intersect_triangle(ray_org, ray_dir, p0, p1, p2, localTHit)
                if hit_res[0]:
                    localTHit = hit_res[1]
                    bestU = hit_res[2]
                    bestV = hit_res[3]
                    hitIndex = offset + j
                    instHit = False

            # Pop next node from stack
            if toVisit == 0:
                break
            toVisit -= 1
            current = Int(stack_ptr[toVisit])
        else:
            # Interior node — test both children, visit nearer first
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)

            var leftNode = bvh2Nodes[leftIdx]
            var rightNode = bvh2Nodes[rightIdx]

            var leftHit = intersect_aabb(
                leftNode.min, leftNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )

            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]

            if leftIsHit and rightIsHit:
                # Both hit — visit nearer first, push farther
                var leftTNear = leftHit[1]
                var rightTNear = rightHit[1]
                if leftTNear <= rightTNear:
                    current = leftIdx
                    stack_ptr[toVisit] = Int32(rightIdx)
                else:
                    current = rightIdx
                    stack_ptr[toVisit] = Int32(leftIdx)
                toVisit += 1
            elif leftIsHit:
                current = leftIdx
            elif rightIsHit:
                current = rightIdx
            else:
                # Neither child hit — pop from stack
                if toVisit == 0:
                    break
                toVisit -= 1
                current = Int(stack_ptr[toVisit])

    if instHit:
        resultPtr[0] = Intersection_C(instHitPrim, localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    elif hitIndex != -1:
        resultPtr[0] = Intersection_C(primIds[hitIndex], localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    else:
        var dummyId = PrimId_C(-1, -1, 0, -1, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)


# Shadow-ray traversal: returns True if anything is hit within tMax (early exit).
@always_inline
def any_hit_bvh2_core(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
) -> Bool:
    var rdir = Vec3f(Float32(1.0) / ray.direction.x, Float32(1.0) / ray.direction.y, Float32(1.0) / ray.direction.z)
    var org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)
    var stack = InlineArray[Int32, 64](fill=Int32(0))
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0
    var ray_org = SIMD[DType.float32, 3](ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = SIMD[DType.float32, 3](ray.direction.x, ray.direction.y, ray.direction.z)
    while True:
        var node = bvh2Nodes[current]
        if node.count > 0:
            var offset = Int(node.offset)
            var count = Int(node.count)
            for j in range(count):
                var prim = primIds[offset + j]
                var mesh_idx: Int
                var base_vidx: Int
                if prim.type == 0:
                    mesh_idx = Int(prim.id1)
                    base_vidx = Int(prim.id2)
                elif prim.type == 1 or prim.type == 2 or prim.type == 3:
                    if prim.id2 == -1:
                        continue
                    mesh_idx = Int(prim.id2 >> 32)
                    base_vidx = Int(prim.id2 & 0xFFFFFFFF) * 3
                elif prim.type == 5:
                    var curve = curves[Int(prim.id1)]
                    if intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, tMax)[0]:
                        return True
                    continue
                elif prim.type == 6:
                    if _traverse_instance_leaf(prim, meshes, blasNodesArr, blasPrimIdsArr, instances, ray_org, ray_dir, tMax)[0]:
                        return True
                    continue
                else:
                    continue
                var mesh = meshes[mesh_idx]
                var v0 = Int(mesh.vertexIndices[base_vidx])
                var v1 = Int(mesh.vertexIndices[base_vidx + 1])
                var v2 = Int(mesh.vertexIndices[base_vidx + 2])
                var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
                var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
                var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])
                if intersect_triangle(ray_org, ray_dir, p0, p1, p2, tMax)[0]:
                    return True
            if toVisit == 0:
                break
            toVisit -= 1
            current = Int(stack_ptr[toVisit])
        else:
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)
            var leftNode = bvh2Nodes[leftIdx]
            var rightNode = bvh2Nodes[rightIdx]
            var leftHit = intersect_aabb(
                leftNode.min, leftNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdir, org,
                nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]
            if leftIsHit and rightIsHit:
                if leftHit[1] <= rightHit[1]:
                    current = leftIdx
                    stack_ptr[toVisit] = Int32(rightIdx)
                else:
                    current = rightIdx
                    stack_ptr[toVisit] = Int32(leftIdx)
                toVisit += 1
            elif leftIsHit:
                current = leftIdx
            elif rightIsHit:
                current = rightIdx
            else:
                if toVisit == 0:
                    break
                toVisit -= 1
                current = Int(stack_ptr[toVisit])
    return False


# ── CPU entry point ─────────────────────────────────────────────────────────

def traverse_bvh2(scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):
    var scene = scenePtr[0]
    var ray = rayPtr[0]
    traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, scene.curves, ray, tMax, resultPtr,
                       scene.blasNodesArr, scene.blasPrimIdsArr, scene.instances)


# ── BVH2 Construction (SAH) ───────────────────────────────────────────

@always_inline
def _bvh_swap(
    widx: UnsafePointer[Int32, MutAnyOrigin],
    wmin: UnsafePointer[Float32, MutAnyOrigin],
    wmax: UnsafePointer[Float32, MutAnyOrigin],
    i: Int, j: Int,
):
    var ti = widx[i]; widx[i] = widx[j]; widx[j] = ti
    for a in range(3):
        var mn = wmin[i*3+a]; wmin[i*3+a] = wmin[j*3+a]; wmin[j*3+a] = mn
        var mx = wmax[i*3+a]; wmax[i*3+a] = wmax[j*3+a]; wmax[j*3+a] = mx

def build_bvh2_node(
    widx: UnsafePointer[Int32, MutAnyOrigin],
    wmin: UnsafePointer[Float32, MutAnyOrigin],
    wmax: UnsafePointer[Float32, MutAnyOrigin],
    start: Int, end: Int,
    out_nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    node_count: UnsafePointer[Int32, MutAnyOrigin],
    prims_per_node: Int,
) -> Int32:
    var my = Int(node_count[0])
    node_count[0] = node_count[0] + 1
    var count = end - start

    var INF = Float32(3.0e38)
    var bminx = INF; var bminy = INF; var bminz = INF
    var bmaxx = -INF; var bmaxy = -INF; var bmaxz = -INF
    var cminx = INF; var cminy = INF; var cminz = INF
    var cmaxx = -INF; var cmaxy = -INF; var cmaxz = -INF
    for i in range(start, end):
        var mnx = wmin[i*3+0]; var mny = wmin[i*3+1]; var mnz = wmin[i*3+2]
        var mxx = wmax[i*3+0]; var mxy = wmax[i*3+1]; var mxz = wmax[i*3+2]
        bminx = min(bminx, mnx); bminy = min(bminy, mny); bminz = min(bminz, mnz)
        bmaxx = max(bmaxx, mxx); bmaxy = max(bmaxy, mxy); bmaxz = max(bmaxz, mxz)
        var cx = Float32(0.5)*(mnx+mxx); var cy = Float32(0.5)*(mny+mxy)
        var cz = Float32(0.5)*(mnz+mxz)
        cminx = min(cminx, cx); cminy = min(cminy, cy); cminz = min(cminz, cz)
        cmaxx = max(cmaxx, cx); cmaxy = max(cmaxy, cy); cmaxz = max(cmaxz, cz)

    # Centroid bounds: pick maximum-extent axis.
    var cex = cmaxx - cminx; var cey = cmaxy - cminy; var cez = cmaxz - cminz
    var dim = 0
    if cey > cex and cey >= cez:
        dim = 1
    elif cez > cex and cez > cey:
        dim = 2
    var cmin_d = cminx if dim == 0 else (cminy if dim == 1 else cminz)
    var cmax_d = cmaxx if dim == 0 else (cmaxy if dim == 1 else cmaxz)

    var dxb = bmaxx - bminx; var dyb = bmaxy - bminy; var dzb = bmaxz - bminz
    if dxb < 0: dxb = 0
    if dyb < 0: dyb = 0
    if dzb < 0: dzb = 0
    var sa = Float32(2.0) * (dxb*dyb + dyb*dzb + dzb*dxb)

    # Leaf when geometry is degenerate or a single primitive.
    if sa == Float32(0.0) or count == 1 or cmax_d == cmin_d:
        out_nodes[my] = BVH2Node(Point3f(bminx, bminy, bminz), Point3f(bmaxx, bmaxy, bmaxz),
                                 Int32(start), Int32(count))
        return Int32(my)

    # Compute the split index `mid` (-1 => fall back to a leaf).
    var mid = -1
    if count <= 2:
        # Order the (at most two) primitives along `dim`; split in the middle.
        if count == 2:
            var c0 = Float32(0.5)*(wmin[start*3+dim] + wmax[start*3+dim])
            var c1 = Float32(0.5)*(wmin[(start+1)*3+dim] + wmax[(start+1)*3+dim])
            if c1 < c0:
                _bvh_swap(widx, wmin, wmax, start, start+1)
        mid = start + count // 2
    else:
        comptime nBuckets = 12
        var bk_cnt = InlineArray[Int32, nBuckets](fill=Int32(0))
        var bk_minx = InlineArray[Float32, nBuckets](fill=INF)
        var bk_miny = InlineArray[Float32, nBuckets](fill=INF)
        var bk_minz = InlineArray[Float32, nBuckets](fill=INF)
        var bk_maxx = InlineArray[Float32, nBuckets](fill=-INF)
        var bk_maxy = InlineArray[Float32, nBuckets](fill=-INF)
        var bk_maxz = InlineArray[Float32, nBuckets](fill=-INF)
        var inv_d = Float32(1.0) / (cmax_d - cmin_d)
        for i in range(start, end):
            var ci = Float32(0.5)*(wmin[i*3+dim] + wmax[i*3+dim])
            var b = Int(Float32(nBuckets) * ((ci - cmin_d) * inv_d))
            if b == nBuckets: b = nBuckets - 1
            if b < 0: b = 0
            bk_cnt[b] += 1
            bk_minx[b] = min(bk_minx[b], wmin[i*3+0])
            bk_miny[b] = min(bk_miny[b], wmin[i*3+1])
            bk_minz[b] = min(bk_minz[b], wmin[i*3+2])
            bk_maxx[b] = max(bk_maxx[b], wmax[i*3+0])
            bk_maxy[b] = max(bk_maxy[b], wmax[i*3+1])
            bk_maxz[b] = max(bk_maxz[b], wmax[i*3+2])

        comptime nSplits = nBuckets - 1
        var costs = InlineArray[Float32, nSplits](fill=Float32(0.0))
        # Prefix pass: cost of the "below" set for each split.
        var cntBelow = 0
        var pminx = INF; var pminy = INF; var pminz = INF
        var pmaxx = -INF; var pmaxy = -INF; var pmaxz = -INF
        for i in range(nSplits):
            pminx = min(pminx, bk_minx[i]); pminy = min(pminy, bk_miny[i])
            pminz = min(pminz, bk_minz[i])
            pmaxx = max(pmaxx, bk_maxx[i]); pmaxy = max(pmaxy, bk_maxy[i])
            pmaxz = max(pmaxz, bk_maxz[i])
            cntBelow += Int(bk_cnt[i])
            var ex = pmaxx - pminx; var ey = pmaxy - pminy; var ez = pmaxz - pminz
            if ex < 0: ex = 0
            if ey < 0: ey = 0
            if ez < 0: ez = 0
            costs[i] += Float32(cntBelow) * Float32(2.0) * (ex*ey + ey*ez + ez*ex)
        # Suffix pass: cost of the "above" set.
        var cntAbove = 0
        var qminx = INF; var qminy = INF; var qminz = INF
        var qmaxx = -INF; var qmaxy = -INF; var qmaxz = -INF
        for i in range(nSplits, 0, -1):
            qminx = min(qminx, bk_minx[i]); qminy = min(qminy, bk_miny[i])
            qminz = min(qminz, bk_minz[i])
            qmaxx = max(qmaxx, bk_maxx[i]); qmaxy = max(qmaxy, bk_maxy[i])
            qmaxz = max(qmaxz, bk_maxz[i])
            cntAbove += Int(bk_cnt[i])
            var ex = qmaxx - qminx; var ey = qmaxy - qminy; var ez = qmaxz - qminz
            if ex < 0: ex = 0
            if ey < 0: ey = 0
            if ez < 0: ez = 0
            costs[i-1] += Float32(cntAbove) * Float32(2.0) * (ex*ey + ey*ez + ez*ex)

        var minBucket = -1
        var minCost = Float32(3.0e38)
        for i in range(nSplits):
            if costs[i] < minCost:
                minCost = costs[i]
                minBucket = i

        var leafCost = Float32(count)
        var splitCost = Float32(0.5) + minCost / sa
        if count > prims_per_node or splitCost < leafCost:
            # Partition: "below" (bucket <= minBucket) first, "above" after.
            var l = start
            for r in range(start, end):
                var ci = Float32(0.5)*(wmin[r*3+dim] + wmax[r*3+dim])
                var b = Int(Float32(nBuckets) * ((ci - cmin_d) * inv_d))
                if b == nBuckets: b = nBuckets - 1
                if b < 0: b = 0
                if b <= minBucket:           # predicate false => "below"
                    if r != l:
                        _bvh_swap(widx, wmin, wmax, l, r)
                    l += 1
            mid = l
            if mid == start or mid == end:
                mid = -1                     # degenerate split => leaf
        # else: leave mid = -1 (leaf)

    if mid < 0:
        out_nodes[my] = BVH2Node(Point3f(bminx, bminy, bminz), Point3f(bmaxx, bmaxy, bmaxz),
                                 Int32(start), Int32(count))
        return Int32(my)

    # Interior node: left child is the next reserved slot (my+1).
    _ = build_bvh2_node(widx, wmin, wmax, start, mid,
                        out_nodes, node_count, prims_per_node)
    var right = build_bvh2_node(widx, wmin, wmax, mid, end,
                                out_nodes, node_count, prims_per_node)
    out_nodes[my] = BVH2Node(Point3f(bminx, bminy, bminz), Point3f(bmaxx, bmaxy, bmaxz),
                             right, Int32(0))
    return Int32(my)


def build_bvh2(
    primBounds: UnsafePointer[Float32, MutAnyOrigin],   # 6 floats per prim
    primCount: Int32,
    outNodes: UnsafePointer[BVH2Node, MutAnyOrigin],     # capacity >= 2*n
    outOrder: UnsafePointer[Int32, MutAnyOrigin],        # capacity >= n
) -> Int32:
    var n = Int(primCount)
    if n <= 0:
        return Int32(0)

    var widx = alloc[Int32](n)
    var wmin = alloc[Float32](n * 3)
    var wmax = alloc[Float32](n * 3)
    for i in range(n):
        widx[i] = Int32(i)
        wmin[i*3+0] = primBounds[i*6+0]
        wmin[i*3+1] = primBounds[i*6+1]
        wmin[i*3+2] = primBounds[i*6+2]
        wmax[i*3+0] = primBounds[i*6+3]
        wmax[i*3+1] = primBounds[i*6+4]
        wmax[i*3+2] = primBounds[i*6+5]

    var node_count = alloc[Int32](1)
    node_count[0] = 0
    _ = build_bvh2_node(widx, wmin, wmax, 0, n, outNodes, node_count, 4)

    for k in range(n):
        outOrder[k] = widx[k]

    var result = node_count[0]
    widx.free(); wmin.free(); wmax.free(); node_count.free()
    return result

# ── Unjittered normals/depth pass, for the denoiser's guide buffers ─────────
# Lives here (not rendering.mojo, where it originated) so bdpt.mojo/sppm.mojo
# can use it too without a circular import: rendering.mojo -> shading.mojo ->
# sppm.mojo already exists, so sppm.mojo (or bdpt.mojo, which imports FROM
# sppm.mojo) importing FROM rendering.mojo would cycle back to itself.
# render_aux_buffers only ever needed BVH/geometry primitives, never
# shading.mojo, so it belongs next to traverse_bvh2_core/test_spheres
# regardless of the cycle -- rendering.mojo re-exposes it via its own import
# so pipeline.mojo's existing `from .rendering import render_aux_buffers`
# still resolves.
def render_aux_buffers(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld:  UnsafePointer[Float32, MutAnyOrigin],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    scene: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
    depth_out:   UnsafePointer[Float32, MutAnyOrigin],
):
    var w  = Int(max_x - min_x)
    var h  = Int(max_y - min_y)
    var n_pixels = w * h
    var sd = scene[0]
    var org = Point3f(cameraToWorld[12], cameraToWorld[13], cameraToWorld[14])
    var isects = alloc[Intersection_C](n_pixels)

    @parameter
    def trace_pixel(i: Int):
        var py = i // w
        var px = i % w
        var filmX = Float32(Int(min_x) + px) + Float32(0.5)
        var filmY = Float32(Int(min_y) + py) + Float32(0.5)

        # rasterToCamera (column-major 4×4), no filter offset
        var cx = rasterToCamera[0]*filmX + rasterToCamera[4]*filmY + rasterToCamera[12]
        var cy = rasterToCamera[1]*filmX + rasterToCamera[5]*filmY + rasterToCamera[13]
        var cz = rasterToCamera[2]*filmX + rasterToCamera[6]*filmY + rasterToCamera[14]
        var cw = rasterToCamera[3]*filmX + rasterToCamera[7]*filmY + rasterToCamera[15]
        if cw != Float32(0.0) and cw != Float32(1.0):
            cx /= cw; cy /= cw; cz /= cw
        var cl = sqrt(cx*cx + cy*cy + cz*cz)
        if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl

        # cameraToWorld rotation (upper-left 3×3)
        var dir = Vec3f(
            cameraToWorld[0]*cx + cameraToWorld[4]*cy + cameraToWorld[8]*cz,
            cameraToWorld[1]*cx + cameraToWorld[5]*cy + cameraToWorld[9]*cz,
            cameraToWorld[2]*cx + cameraToWorld[6]*cy + cameraToWorld[10]*cz,
        )
        var dl = dir.length()
        if dl > Float32(0): dir = dir / dl

        var ray = Ray_C(org, dir)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), isects + i,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if Int(sd.sphereCount) > 0:
            test_spheres(sd.spheres, Int(sd.sphereCount), ray, isects + i)

        var normal = Vec3f(Float32(0), Float32(0), Float32(1))   # background
        var d  = Float32(1e38)

        if isects[i].hit != Int8(0):
            d = isects[i].tHit
            var typ = Int(isects[i].primId.type)
            if typ == 4:
                # Sphere: normal = normalize(hit_point - center)
                var si  = Int(isects[i].primId.id1)
                normal = sphere_outward_normal(org + dir*d, sd.spheres[si].center)
            elif typ == 5:
                # Native curve: reconstruct the geometric normal the same way
                # shade_hair does (shading.mojo) — h/v come straight from
                # intersect_curve via Intersection_C.u/.v, no tessellated mesh.
                var curve = sd.curves[Int(isects[i].primId.id1)]
                var h = max(Float32(-0.99), min(Float32(0.99), isects[i].u))
                var piece = min(Int(curve.n_pieces) - 1, max(0, Int(isects[i].v * Float32(curve.n_pieces))))
                var (q0, q1, _, _) = curve_piece_endpoints(curve, piece)
                var seg_axis = q1 - q0
                var seg_len = sqrt(dot(seg_axis, seg_axis))
                var tangent: SIMD[DType.float32, 3]
                if seg_len > Float32(1e-8):
                    tangent = seg_axis * (Float32(1.0) / seg_len)
                else:
                    tangent = SIMD[DType.float32, 3](Float32(1), Float32(0), Float32(0))
                var n_perp = _curve_perp_axis(tangent)
                var b_perp0 = cross(tangent, n_perp)
                var geo = n_perp * h + b_perp0 * sqrt(max(Float32(0.0), Float32(1.0) - h*h))
                var gl = sqrt(dot(geo, geo))
                if gl > Float32(0):
                    normal = Vec3f(geo[0] / gl, geo[1] / gl, geo[2] / gl)
            elif typ == 0 or typ == 1 or typ == 2 or typ == 3:
                # Triangle: geometric normal from edge cross product
                var mesh_idx: Int
                var base_vidx: Int
                if typ == 0:
                    mesh_idx  = Int(isects[i].primId.id1)
                    base_vidx = Int(isects[i].primId.id2)
                else:
                    mesh_idx  = Int(isects[i].primId.id2 >> 32)
                    base_vidx = Int(isects[i].primId.id2 & 0xFFFFFFFF) * 3
                var mesh = sd.meshes[mesh_idx]
                var vi0 = Int(mesh.vertexIndices[base_vidx])
                var vi1 = Int(mesh.vertexIndices[base_vidx + 1])
                var vi2 = Int(mesh.vertexIndices[base_vidx + 2])
                var p0 = Point3f(mesh.points[vi0*4], mesh.points[vi0*4+1], mesh.points[vi0*4+2])
                var p1 = Point3f(mesh.points[vi1*4], mesh.points[vi1*4+1], mesh.points[vi1*4+2])
                var p2 = Point3f(mesh.points[vi2*4], mesh.points[vi2*4+1], mesh.points[vi2*4+2])
                var e1 = p1 - p0; var e2 = p2 - p0
                normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
                var nl = normal.length()
                if nl > Float32(0): normal = normal / nl
            # else: unrecognized hit type (shouldn't occur for a real
            # Intersection_C — type==6 only ever exists as a transient BVH
            # leaf marker, resolved into type 0-3 with instanceIdx set before
            # traversal returns) — leave `normal` at its background default
            # rather than guessing.
            # Flip to face incoming ray
            if normal.dot(-dir) < Float32(0):
                normal = -normal

        normals_out[i*3 + 0] = normal.x
        normals_out[i*3 + 1] = normal.y
        normals_out[i*3 + 2] = normal.z
        depth_out[i] = d

    parallelize[trace_pixel](n_pixels)
    isects.free()
