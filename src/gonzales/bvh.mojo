from std.memory import alloc
from std.math import sqrt, cos, sin, max, min
from .geometry import Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, Curve_C, intersect_curve, CURVE_DEFER_K, DistantLight_C, PointLight_C, InfiniteLight_C, dot, cross, intersect_triangle, PathState_C, TileResult_C, Point3f, Point2f, Vec3f, Frame, RGB, Medium_C, MediumInterface_C, Grid_C, LightSampler_C, Instance_C, PI

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
    textures, point lights, grids, and the light sampler CDF are never
    touched by either module's code paths, so they stay dangling/zero-count,
    same convention traverse_bvh2_core's own optional instancing args
    already use. distant/infinite lights default to the same dangling
    convention for backward compatibility, but callers that need
    infinite/distant-light NEE or light-path emission (both bdpt.mojo and
    sppm.mojo now do, see [[project_priority_backlog]] item 1) should pass
    the real device buffers/counts explicitly."""
    return SceneDescriptor2_C(
        bvh2Nodes=bvh2Nodes, primIds=primIds,
        meshes=meshes, meshCount=meshCount,
        materials=materials, materialCount=materialCount,
        areaLights=areaLights, areaLightCount=areaLightCount,
        textures=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textureCount=Int64(0),
        distantLights=distantLights,
        distantLightCount=distantLightCount,
        pointLights=UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(),
        pointLightCount=Int64(0),
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
def _equal_area_square_to_sphere_bvh(uv: Point2f) -> Vec3f:
    """Duplicate of shading.mojo's _equal_area_square_to_sphere (PBRT v4's
    equal-area octahedral mapping, Clarberg 2008) — see this section's
    docstring for why it's duplicated rather than imported."""
    var uu = Float32(2) * uv.x - Float32(1)
    var vv = Float32(2) * uv.y - Float32(1)
    var up = uu if uu >= Float32(0) else -uu
    var vp = vv if vv >= Float32(0) else -vv
    var signed_dist = Float32(1) - (up + vp)
    var d = signed_dist if signed_dist >= Float32(0) else -signed_dist
    var r = Float32(1) - d
    var phi: Float32
    if r == Float32(0):
        phi = Float32(0)
    elif up >= vp:
        phi = (vp / r) * PI / Float32(4)
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
    return Vec3f(cp * xy_scale, sp * xy_scale, z)

@always_inline
def _lower_bound_bvh(arr: UnsafePointer[Float32, MutAnyOrigin], lo: Int, hi: Int, val: Float32) -> Int:
    """Duplicate of shading.mojo's _lower_bound: first index i in [lo, hi)
    s.t. arr[i] >= val; returns hi if all < val."""
    var l = lo; var h = hi
    while l < h:
        var mid = (l + h) // 2
        if arr[mid] < val:
            l = mid + 1
        else:
            h = mid
    return l

@always_inline
def _sample_infinite_light_dir(
    ilight: InfiniteLight_C,
    u: Point2f,
) -> Tuple[Vec3f, RGB, Float32]:
    """Sample an emission direction from an environment (infinite) light,
    returning (world-space direction, radiance there, solid-angle pdf).
    Used for BOTH light-path/photon emission (bdpt.mojo/sppm.mojo) and NEE
    toward the light (same distribution works for both — the only
    difference is which end of the ray you start from). Mirrors
    shading.mojo's _nee_infinite_light CDF-importance-sampling block."""
    var w2l = ilight.world_to_light
    if ilight.tex_idx >= Int32(0) and Int(ilight.pixels_ptr) > 1 and Int(ilight.cdf_ptr) > 1 and ilight.cdf_w > Int32(0):
        var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
        var row_idx = _lower_bound_bvh(ilight.cdf_ptr, 0, ih, u.x)
        row_idx = min(row_idx, ih - 1)
        var dp_row = ilight.cdf_ptr[row_idx + 1] - ilight.cdf_ptr[row_idx]
        var cond_base = (ih + 1) + row_idx * (iw + 1)
        var col_idx = _lower_bound_bvh(ilight.cdf_ptr, cond_base, cond_base + iw, u.y) - cond_base
        col_idx = min(col_idx, iw - 1)
        var dp_col = ilight.cdf_ptr[cond_base + col_idx + 1] - ilight.cdf_ptr[cond_base + col_idx]

        var sample_u = (Float32(col_idx) + Float32(0.5)) / Float32(iw)
        var sample_v = (Float32(row_idx) + Float32(0.5)) / Float32(ih)
        var local_d = _equal_area_square_to_sphere_bvh(Point2f(sample_u, sample_v))

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
    else:
        # No texture: uniform sphere sampling, constant radiance.
        var cosT = Float32(2) * u.x - Float32(1)
        var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
        var phi = Float32(2) * PI * u.y
        var dir = Vec3f(sinT * cos(phi), sinT * sin(phi), cosT)
        return (dir, ilight.scale, Float32(1) / (Float32(4) * PI))

@always_inline
def _equal_area_sphere_to_square_bvh(dir: Vec3f) -> Point2f:
    """Duplicate of shading.mojo's _equal_area_sphere_to_square (inverse of
    _equal_area_square_to_sphere_bvh) — see this section's docstring for why
    it's duplicated rather than imported."""
    var x = dir.x if dir.x >= Float32(0) else -dir.x
    var y = dir.y if dir.y >= Float32(0) else -dir.y
    var z = dir.z if dir.z >= Float32(0) else -dir.z
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
    if dir.z < Float32(0):
        var tmp = u
        u = Float32(1) - v
        v = Float32(1) - tmp
    if dir.x < Float32(0): u = -u
    if dir.y < Float32(0): v = -v
    u = Float32(0.5) * (u + Float32(1))
    v = Float32(0.5) * (v + Float32(1))
    return Point2f(u, v)

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
    if ilight.tex_idx < Int32(0) or Int(ilight.pixels_ptr) <= 1 or ilight.cdf_w <= Int32(0):
        return (ilight.scale, Float32(1) / (Float32(4) * PI))
    var w2l = ilight.world_to_light
    var local_dir = Vec3f(
        w2l[0]*dir_world.x + w2l[4]*dir_world.y + w2l[8]*dir_world.z,
        w2l[1]*dir_world.x + w2l[5]*dir_world.y + w2l[9]*dir_world.z,
        w2l[2]*dir_world.x + w2l[6]*dir_world.y + w2l[10]*dir_world.z,
    )
    var uv = _equal_area_sphere_to_square_bvh(local_dir)
    var iw = Int(ilight.cdf_w); var ih = Int(ilight.cdf_h)
    var px = min(iw - 1, max(0, Int(uv.x * Float32(iw))))
    var py = min(ih - 1, max(0, Int(uv.y * Float32(ih))))
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
