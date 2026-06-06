from std.memory import alloc
from std.math import sqrt
from .geometry import Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C, dot, cross, intersect_triangle, PathState_C, TileResult_C, PixelSample_C, Point3f, Vec3f, Medium_C, MediumInterface_C

# ── BVH2 Compact Nodes (32 bytes per node, 1 cache line) ──────────────────────
# Layout: Point3f min (12 B) + Point3f max (12 B) + Int32 offset (4 B) + Int32 count (4 B) = 32 B

@fieldwise_init
struct BVH2Node(TrivialRegisterPassable):
    var min: Point3f        # AABB minimum corner
    var max: Point3f        # AABB maximum corner
    var offset: Int32       # interior: right child index, leaf: primIds offset
    var count: Int32        # 0 = interior, >0 = leaf primitive count

@always_inline
fn intersect_aabb(
    bmin: Point3f, bmax: Point3f,
    rdirX: Float32, rdirY: Float32, rdirZ: Float32,
    orgRdirX: Float32, orgRdirY: Float32, orgRdirZ: Float32,
    nearXIsMin: Bool, nearYIsMin: Bool, nearZIsMin: Bool,
    tMax: Float32
) -> Tuple[Bool, Float32]:
    var nearX = bmin.x if nearXIsMin else bmax.x
    var farX  = bmax.x if nearXIsMin else bmin.x
    var nearY = bmin.y if nearYIsMin else bmax.y
    var farY  = bmax.y if nearYIsMin else bmin.y
    var nearZ = bmin.z if nearZIsMin else bmax.z
    var farZ  = bmax.z if nearZIsMin else bmin.z

    var tNearX = nearX * rdirX - orgRdirX
    var tNearY = nearY * rdirY - orgRdirY
    var tNearZ = nearZ * rdirZ - orgRdirZ

    var tFarX = farX * rdirX - orgRdirX
    var tFarY = farY * rdirY - orgRdirY
    var tFarZ = farZ * rdirZ - orgRdirZ

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
    var mediums: UnsafePointer[Medium_C, MutAnyOrigin]
    var mediumCount: Int64
    var mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin]
    var mediumIfaceCount: Int64

# ── Analytical sphere intersection ────────────────────────────────────────────

@always_inline
fn ray_sphere_hit(center: Point3f, radius: Float32,
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
fn test_spheres(
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

# ── Unified traversal core (CPU + GPU) ────────────────────────────────────────

@always_inline
fn traverse_bvh2_core(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
    resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin],
):

    var rdirX = Float32(1.0) / ray.direction.x
    var rdirY = Float32(1.0) / ray.direction.y
    var rdirZ = Float32(1.0) / ray.direction.z

    var orgRdirX = ray.origin.x * rdirX
    var orgRdirY = ray.origin.y * rdirY
    var orgRdirZ = ray.origin.z * rdirZ

    var nearXIsMin = rdirX >= Float32(0.0)
    var nearYIsMin = rdirY >= Float32(0.0)
    var nearZIsMin = rdirZ >= Float32(0.0)

    var hitIndex: Int = -1
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0

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
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
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

    if hitIndex != -1:
        resultPtr[0] = Intersection_C(primIds[hitIndex], localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    else:
        var dummyId = PrimId_C(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)


# Shadow-ray traversal: returns True if anything is hit within tMax (early exit).
@always_inline
fn any_hit_bvh2_core(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
) -> Bool:
    var rdirX = Float32(1.0) / ray.direction.x
    var rdirY = Float32(1.0) / ray.direction.y
    var rdirZ = Float32(1.0) / ray.direction.z
    var orgRdirX = ray.origin.x * rdirX
    var orgRdirY = ray.origin.y * rdirY
    var orgRdirZ = ray.origin.z * rdirZ
    var nearXIsMin = rdirX >= Float32(0.0)
    var nearYIsMin = rdirY >= Float32(0.0)
    var nearZIsMin = rdirZ >= Float32(0.0)
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
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
                nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var rightHit = intersect_aabb(
                rightNode.min, rightNode.max,
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
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

@export
fn mojo_traverse_bvh2(scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):
    var scene = scenePtr[0]
    var ray = rayPtr[0]
    traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, ray, tMax, resultPtr)


# ── CPU batch entry point (sequential loop) ───────────────────────────────────

@export
fn mojo_cpu_traverse_batch(
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    rays: UnsafePointer[Ray_C, MutAnyOrigin],
    tMaxValues: UnsafePointer[Float32, MutAnyOrigin],
    count: Int64,
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
):
    var scene = scenePtr[0]
    var n = Int(count)
    for tid in range(n):
        traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes,
                          rays[tid], tMaxValues[tid], results + tid)


# ── BVH2 Construction (SAH) ───────────────────────────────────────────

@always_inline
fn _bvh_swap(
    widx: UnsafePointer[Int32, MutAnyOrigin],
    wmin: UnsafePointer[Float32, MutAnyOrigin],
    wmax: UnsafePointer[Float32, MutAnyOrigin],
    i: Int, j: Int,
):
    var ti = widx[i]; widx[i] = widx[j]; widx[j] = ti
    for a in range(3):
        var mn = wmin[i*3+a]; wmin[i*3+a] = wmin[j*3+a]; wmin[j*3+a] = mn
        var mx = wmax[i*3+a]; wmax[i*3+a] = wmax[j*3+a]; wmax[j*3+a] = mx

fn build_bvh2_node(
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


@export
def mojo_build_bvh2(
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
