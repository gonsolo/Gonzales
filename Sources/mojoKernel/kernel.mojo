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
struct BoundingHierarchyNode(TrivialRegisterPassable):
    var pMinX: SIMD[DType.float32, 8]
    var pMaxX: SIMD[DType.float32, 8]
    var pMinY: SIMD[DType.float32, 8]
    var pMaxY: SIMD[DType.float32, 8]
    var pMinZ: SIMD[DType.float32, 8]
    var pMaxZ: SIMD[DType.float32, 8]
    var childNodes: SIMD[DType.int32, 8]
    var primitiveOffsets: SIMD[DType.int32, 8]
    var primitiveCounts: SIMD[DType.int32, 8]

    @always_inline
    fn intersect8(
        self,
        rdirX: SIMD[DType.float32, 8],
        rdirY: SIMD[DType.float32, 8],
        rdirZ: SIMD[DType.float32, 8],
        orgRdirX: SIMD[DType.float32, 8],
        orgRdirY: SIMD[DType.float32, 8],
        orgRdirZ: SIMD[DType.float32, 8],
        nearXIsMin: Bool,
        nearYIsMin: Bool,
        nearZIsMin: Bool,
        tHit: Float32
    ) -> SIMD[DType.bool, 8]:

        var nearX = self.pMinX if nearXIsMin else self.pMaxX
        var farX = self.pMaxX if nearXIsMin else self.pMinX
        var nearY = self.pMinY if nearYIsMin else self.pMaxY
        var farY = self.pMaxY if nearYIsMin else self.pMinY
        var nearZ = self.pMinZ if nearZIsMin else self.pMaxZ
        var farZ = self.pMaxZ if nearZIsMin else self.pMinZ

        var tNearX = nearX * rdirX - orgRdirX
        var tNearY = nearY * rdirY - orgRdirY
        var tNearZ = nearZ * rdirZ - orgRdirZ

        var tFarX = farX * rdirX - orgRdirX
        var tFarY = farY * rdirY - orgRdirY
        var tFarZ = farZ * rdirZ - orgRdirZ

        # tNear = max(tNearX, tNearY, tNearZ, 0) - NaN-safe using element-wise comparisons
        # When comparison involves NaN, .gt()/.lt() return False, keeping the original value
        var zero = SIMD[DType.float32, 8](0.0)
        var tNear = tNearY.gt(tNearX).select(tNearY, tNearX)
        tNear = tNearZ.gt(tNear).select(tNearZ, tNear)
        tNear = zero.gt(tNear).select(zero, tNear)

        # tFar = min(tFarX, tFarY, tFarZ, tHit) * gamma - NaN-safe
        var tHitV = SIMD[DType.float32, 8](tHit)
        var tFar = tFarY.lt(tFarX).select(tFarY, tFarX)
        tFar = tFarZ.lt(tFar).select(tFarZ, tFar)
        tFar = tHitV.lt(tFar).select(tHitV, tFar)

        var safeGamma = SIMD[DType.float32, 8](1.0000003)
        tFar = tFar * safeGamma

        return tNear.le(tFar)

@fieldwise_init
struct SceneDescriptor_C(TrivialRegisterPassable):
    var bvhNodes: UnsafePointer[BoundingHierarchyNode, MutAnyOrigin]
    var primIds: UnsafePointer[PrimId_C, MutAnyOrigin]
    var meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin]
    var meshCount: Int64

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

@export
fn mojo_test_intersect(bvhNodes: UnsafePointer[BoundingHierarchyNode, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32) -> Int32:
    """Run intersect8 on root node and return bitmask result."""
    var ray = rayPtr[0]
    var node = bvhNodes[0]

    var rdirXs = Float32(1.0) / ray.dirX
    var rdirYs = Float32(1.0) / ray.dirY
    var rdirZs = Float32(1.0) / ray.dirZ
    
    var rdirX = SIMD[DType.float32, 8](rdirXs)
    var rdirY = SIMD[DType.float32, 8](rdirYs)
    var rdirZ = SIMD[DType.float32, 8](rdirZs)
    var orgRdirX = SIMD[DType.float32, 8](ray.orgX * rdirXs)
    var orgRdirY = SIMD[DType.float32, 8](ray.orgY * rdirYs)
    var orgRdirZ = SIMD[DType.float32, 8](ray.orgZ * rdirZs)
    var nearXIsMin = rdirXs >= Float32(0.0)
    var nearYIsMin = rdirYs >= Float32(0.0)
    var nearZIsMin = rdirZs >= Float32(0.0)

    var mask = node.intersect8(
        rdirX, rdirY, rdirZ,
        orgRdirX, orgRdirY, orgRdirZ,
        nearXIsMin, nearYIsMin, nearZIsMin,
        tMax
    )
    
    var result: Int32 = 0
    for i in range(8):
        if mask[i]:
            result |= Int32(1 << i)
    return result

@export
fn mojo_traverse(scenePtr: UnsafePointer[SceneDescriptor_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):

    var scene = scenePtr[0]
    var ray = rayPtr[0]

    var rdirX = SIMD[DType.float32, 8](1.0 / ray.dirX)
    var rdirY = SIMD[DType.float32, 8](1.0 / ray.dirY)
    var rdirZ = SIMD[DType.float32, 8](1.0 / ray.dirZ)
    
    var orgRdirX = SIMD[DType.float32, 8](ray.orgX * (1.0 / ray.dirX))
    var orgRdirY = SIMD[DType.float32, 8](ray.orgY * (1.0 / ray.dirY))
    var orgRdirZ = SIMD[DType.float32, 8](ray.orgZ * (1.0 / ray.dirZ))
    
    var nearXIsMin = (1.0 / ray.dirX) >= 0.0
    var nearYIsMin = (1.0 / ray.dirY) >= 0.0
    var nearZIsMin = (1.0 / ray.dirZ) >= 0.0

    var hitIndex: Int = -1
    var localTHit = tMax
    var bestU: Float32 = 0.0
    var bestV: Float32 = 0.0

    var stack = InlineArray[Int, 128](fill=0)
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0

    var ray_org = SIMD[DType.float32, 3](ray.orgX, ray.orgY, ray.orgZ)
    var ray_dir = SIMD[DType.float32, 3](ray.dirX, ray.dirY, ray.dirZ)

    while True:
        var node = scene.bvhNodes[current]
        var mask = node.intersect8(
            rdirX, rdirY, rdirZ,
            orgRdirX, orgRdirY, orgRdirZ,
            nearXIsMin, nearYIsMin, nearZIsMin,
            localTHit
        )

        for i in range(8):
            if mask[i]:
                
                var count = Int(node.primitiveCounts[i])
                if count > 0: # leaf node
                    var offset = Int(node.primitiveOffsets[i])
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
                            
                            # Points are stored as SIMD4<Float> in Swift, so offset by * 4
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
                else:
                    var childIdx = Int(node.childNodes[i])
                    if childIdx >= 0:
                        stack_ptr[toVisit] = childIdx
                        toVisit += 1

        if toVisit == 0:
            break
        toVisit -= 1
        current = stack_ptr[toVisit]

    if hitIndex != -1:
        resultPtr[0] = Intersection_C(scene.primIds[hitIndex], localTHit, bestU, bestV, Int8(1), 0, 0, 0)
    else:
        var dummyId = PrimId_C(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)


