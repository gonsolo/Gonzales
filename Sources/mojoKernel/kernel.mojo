from std.ffi import external_call

@fieldwise_init
struct PrimId_C(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8

@fieldwise_init
struct TriangleMesh_C(TrivialRegisterPassable):
    var points: UnsafePointer[Float32, MutAnyOrigin]
    var faceIndices: UnsafePointer[Int64, MutAnyOrigin]
    var vertexIndices: UnsafePointer[Int64, MutAnyOrigin]

@fieldwise_init
struct Ray_C(TrivialRegisterPassable):
    var orgX: Float32
    var orgY: Float32
    var orgZ: Float32
    var dirX: Float32
    var dirY: Float32
    var dirZ: Float32

@fieldwise_init
struct Intersection_C(TrivialRegisterPassable):
    var primId: PrimId_C
    var tHit: Float32
    var u: Float32
    var v: Float32
    var hit: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8

@always_inline
fn cross(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> SIMD[DType.float32, 3]:
    var a_yzx = SIMD[DType.float32, 3](a[1], a[2], a[0])
    var b_zxy = SIMD[DType.float32, 3](b[2], b[0], b[1])
    var a_zxy = SIMD[DType.float32, 3](a[2], a[0], a[1])
    var b_yzx = SIMD[DType.float32, 3](b[1], b[2], b[0])
    return a_yzx * b_zxy - a_zxy * b_yzx

@always_inline
fn dot(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Float32:
    var prod = a * b
    return prod[0] + prod[1] + prod[2]

@always_inline
fn intersect_triangle(
    ray_org: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    p0: SIMD[DType.float32, 3],
    p1: SIMD[DType.float32, 3],
    p2: SIMD[DType.float32, 3],
    tMax: Float32
) -> Tuple[Bool, Float32, Float32, Float32]:
    var e1 = p1 - p0
    var e2 = p2 - p0
    var pvec = cross(ray_dir, e2)
    var det = dot(e1, pvec)
    
    if det > -0.0000001 and det < 0.0000001:
        return (False, tMax, 0.0, 0.0)
        
    var invDet = 1.0 / det
    var tvec = ray_org - p0
    var u = dot(tvec, pvec) * invDet
    
    if u < 0.0 or u > 1.0:
        return (False, tMax, 0.0, 0.0)
        
    var qvec = cross(tvec, e1)
    var v = dot(ray_dir, qvec) * invDet
    
    if v < 0.0 or u + v > 1.0:
        return (False, tMax, 0.0, 0.0)
        
    var t = dot(e2, qvec) * invDet
    if t <= 0.0 or t > tMax:
        return (False, tMax, 0.0, 0.0)
        
    return (True, t, u, v)

# ── BVH2 Compact Nodes (32 bytes per node, 1 cache line) ──────────────────────

@fieldwise_init
struct BVH2Node(TrivialRegisterPassable):
    var boundsMinX: Float32
    var boundsMinY: Float32
    var boundsMinZ: Float32
    var boundsMaxX: Float32
    var boundsMaxY: Float32
    var boundsMaxZ: Float32
    var offset: Int32       # interior: right child index, leaf: primIds offset
    var count: Int32        # 0 = interior, >0 = leaf primitive count

@always_inline
fn intersect_aabb(
    boundsMinX: Float32, boundsMinY: Float32, boundsMinZ: Float32,
    boundsMaxX: Float32, boundsMaxY: Float32, boundsMaxZ: Float32,
    rdirX: Float32, rdirY: Float32, rdirZ: Float32,
    orgRdirX: Float32, orgRdirY: Float32, orgRdirZ: Float32,
    nearXIsMin: Bool, nearYIsMin: Bool, nearZIsMin: Bool,
    tMax: Float32
) -> Tuple[Bool, Float32]:
    var nearX = boundsMinX if nearXIsMin else boundsMaxX
    var farX  = boundsMaxX if nearXIsMin else boundsMinX
    var nearY = boundsMinY if nearYIsMin else boundsMaxY
    var farY  = boundsMaxY if nearYIsMin else boundsMinY
    var nearZ = boundsMinZ if nearZIsMin else boundsMaxZ
    var farZ  = boundsMaxZ if nearZIsMin else boundsMinZ

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

@export
fn mojo_traverse_bvh2(scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):

    var scene = scenePtr[0]
    var ray = rayPtr[0]

    var rdirX = Float32(1.0) / ray.dirX
    var rdirY = Float32(1.0) / ray.dirY
    var rdirZ = Float32(1.0) / ray.dirZ

    var orgRdirX = ray.orgX * rdirX
    var orgRdirY = ray.orgY * rdirY
    var orgRdirZ = ray.orgZ * rdirZ

    var nearXIsMin = rdirX >= Float32(0.0)
    var nearYIsMin = rdirY >= Float32(0.0)
    var nearZIsMin = rdirZ >= Float32(0.0)

    var hitIndex: Int = -1
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0

    var stack = InlineArray[Int, 64](fill=0)
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0

    var ray_org = SIMD[DType.float32, 3](ray.orgX, ray.orgY, ray.orgZ)
    var ray_dir = SIMD[DType.float32, 3](ray.dirX, ray.dirY, ray.dirZ)

    while True:
        var node = scene.bvh2Nodes[current]

        if node.count > 0:
            # Leaf node — intersect primitives
            var offset = Int(node.offset)
            var count = Int(node.count)
            for j in range(count):
                var prim = scene.primIds[offset + j]
                if prim.type == 0 or prim.type == 1 or prim.type == 2:
                    var mesh_idx = Int(prim.id1)
                    var tri_idx = Int(prim.id2)

                    if prim.type == 1 or prim.type == 2:
                        mesh_idx = Int(prim.id2 >> 32)
                        tri_idx = Int(prim.id2 & 0xFFFFFFFF)

                    var mesh = scene.meshes[mesh_idx]
                    var v0_idx = Int(mesh.vertexIndices[tri_idx * 3])
                    var v1_idx = Int(mesh.vertexIndices[tri_idx * 3 + 1])
                    var v2_idx = Int(mesh.vertexIndices[tri_idx * 3 + 2])

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
            current = stack_ptr[toVisit]
        else:
            # Interior node — test both children, visit nearer first
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)

            var leftNode = scene.bvh2Nodes[leftIdx]
            var rightNode = scene.bvh2Nodes[rightIdx]

            var leftHit = intersect_aabb(
                leftNode.boundsMinX, leftNode.boundsMinY, leftNode.boundsMinZ,
                leftNode.boundsMaxX, leftNode.boundsMaxY, leftNode.boundsMaxZ,
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
                nearXIsMin, nearYIsMin, nearZIsMin, localTHit
            )
            var rightHit = intersect_aabb(
                rightNode.boundsMinX, rightNode.boundsMinY, rightNode.boundsMinZ,
                rightNode.boundsMaxX, rightNode.boundsMaxY, rightNode.boundsMaxZ,
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
                    stack_ptr[toVisit] = rightIdx
                else:
                    current = rightIdx
                    stack_ptr[toVisit] = leftIdx
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
                current = stack_ptr[toVisit]

    if hitIndex != -1:
        resultPtr[0] = Intersection_C(scene.primIds[hitIndex], localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    else:
        var dummyId = PrimId_C(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)

