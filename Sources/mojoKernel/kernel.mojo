from std.ffi import external_call
from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt, cos, sin, log, exp
from std.memory import alloc

@fieldwise_init
struct PrimId_C(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var materialIndex: Int64
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8

@fieldwise_init
struct Material_C(TrivialRegisterPassable):
    var type: Int8
    var _pad0: Int8
    var _pad1: Int8
    var _pad2: Int8
    var albedoR: Float32
    var albedoG: Float32
    var albedoB: Float32
    var emissionR: Float32
    var emissionG: Float32
    var emissionB: Float32

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

@fieldwise_init
struct PathState_C(TrivialRegisterPassable):
    var ray: Ray_C
    var throughputR: Float32
    var throughputG: Float32
    var throughputB: Float32
    var estimateR: Float32
    var estimateG: Float32
    var estimateB: Float32
    var albedoR: Float32
    var albedoG: Float32
    var albedoB: Float32
    var bounce: Int32
    var pcgState: UInt64
    var pcgInc: UInt64
    var active: Int8
    var _pad1: Int8
    var _pad2: Int8
    var _pad3: Int8
    var _pad4: Int8
    var _pad5: Int8
    var _pad6: Int8
    var _pad7: Int8

@fieldwise_init
struct AreaLight_C(TrivialRegisterPassable):
    var meshIdx: Int32
    var triBaseVidx: Int32
    var emissionR: Float32
    var emissionG: Float32
    var emissionB: Float32
    var _pad: Int32

@fieldwise_init
struct PixelSample_C(TrivialRegisterPassable):
    var filmX: Float32
    var filmY: Float32
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32
    var _pad: Int32
    var pcgState: UInt64
    var pcgInc: UInt64

@fieldwise_init
struct TileResult_C(TrivialRegisterPassable):
    var estimateR: Float32
    var estimateG: Float32
    var estimateB: Float32
    var albedoR: Float32
    var albedoG: Float32
    var albedoB: Float32
    var filterWeight: Float32
    var pixelX: Int32
    var pixelY: Int32

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

# ── Random Number Generation ────────────────────────────────────────

struct PCG32:
    var state: UInt64
    var inc: UInt64

    fn __init__(out self, initstate: UInt64, initseq: UInt64):
        self.state = 0
        self.inc = (initseq << 1) | 1
        _ = self.next_uint()
        self.state += initstate
        _ = self.next_uint()

    fn next_uint(mut self) -> UInt32:
        var oldstate = self.state
        self.state = oldstate * 6364136223846793005 + self.inc
        var xorshifted = UInt32(((oldstate >> 18) ^ oldstate) >> 27)
        var rot = UInt32(oldstate >> 59)
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    fn next_float(mut self) -> Float32:
        return Float32(self.next_uint() >> 8) * (1.0 / 16777216.0)

# ── Ray Intersection Geometry ────────────────────────────────────────

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
    var materials: UnsafePointer[Material_C, MutAnyOrigin]
    var materialCount: Int64
    var areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin]
    var areaLightCount: Int64

# ── Unified traversal core (CPU + GPU) ────────────────────────────────────────
#
# This function contains the BVH traversal logic that is shared between the
# CPU @export path and the GPU kernel. Both call this with raw pointers to
# scene data, regardless of whether that data lives in host or device memory.

@always_inline
fn traverse_bvh2_core(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    ray: Ray_C,
    tMax: Float32,
    resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin],
):

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
            current = stack_ptr[toVisit]
        else:
            # Interior node — test both children, visit nearer first
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)

            var leftNode = bvh2Nodes[leftIdx]
            var rightNode = bvh2Nodes[rightIdx]

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
    var rdirX = Float32(1.0) / ray.dirX
    var rdirY = Float32(1.0) / ray.dirY
    var rdirZ = Float32(1.0) / ray.dirZ
    var orgRdirX = ray.orgX * rdirX
    var orgRdirY = ray.orgY * rdirY
    var orgRdirZ = ray.orgZ * rdirZ
    var nearXIsMin = rdirX >= Float32(0.0)
    var nearYIsMin = rdirY >= Float32(0.0)
    var nearZIsMin = rdirZ >= Float32(0.0)
    var stack = InlineArray[Int, 64](fill=0)
    var stack_ptr = stack.unsafe_ptr()
    var toVisit = 0
    var current = 0
    var ray_org = SIMD[DType.float32, 3](ray.orgX, ray.orgY, ray.orgZ)
    var ray_dir = SIMD[DType.float32, 3](ray.dirX, ray.dirY, ray.dirZ)
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
            current = stack_ptr[toVisit]
        else:
            var leftIdx = current + 1
            var rightIdx = Int(node.offset)
            var leftNode = bvh2Nodes[leftIdx]
            var rightNode = bvh2Nodes[rightIdx]
            var leftHit = intersect_aabb(
                leftNode.boundsMinX, leftNode.boundsMinY, leftNode.boundsMinZ,
                leftNode.boundsMaxX, leftNode.boundsMaxY, leftNode.boundsMaxZ,
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
                nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var rightHit = intersect_aabb(
                rightNode.boundsMinX, rightNode.boundsMinY, rightNode.boundsMinZ,
                rightNode.boundsMaxX, rightNode.boundsMaxY, rightNode.boundsMaxZ,
                rdirX, rdirY, rdirZ, orgRdirX, orgRdirY, orgRdirZ,
                nearXIsMin, nearYIsMin, nearZIsMin, tMax)
            var leftIsHit = leftHit[0]
            var rightIsHit = rightHit[0]
            if leftIsHit and rightIsHit:
                if leftHit[1] <= rightHit[1]:
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
                if toVisit == 0:
                    break
                toVisit -= 1
                current = stack_ptr[toVisit]
    return False


# ── CPU entry point (called from Swift via C FFI) ─────────────────────────────

@export
fn mojo_traverse_bvh2(scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):
    var scene = scenePtr[0]
    var ray = rayPtr[0]
    traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, ray, tMax, resultPtr)


# ── CPU batch entry point (sequential loop, parallelism via Swift tasks) ───────

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


@export
fn mojo_cpu_shade_batch(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    count: Int64,
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
):
    var n = Int(count)
    for tid in range(n):
        shade_core(paths, intersections, meshes, materials, tid)


# ── GPU support ───────────────────────────────────────────────────────────────

@export
fn mojo_gpu_available() -> Bool:
    return has_accelerator()

# GPU scene handle — holds DeviceContext and device-resident scene buffers.
# Allocated on the heap, returned to Swift as an opaque pointer.
@fieldwise_init
struct GpuSceneHandle(Movable):
    var ctx: DeviceContext
    var bvh2Nodes_buf: DeviceBuffer[DType.uint8]
    var primIds_buf: DeviceBuffer[DType.uint8]
    # For meshes: we store a flat buffer of TriangleMesh_C structs, but the
    # points/indices pointers inside them point to device memory.
    var meshes_buf: DeviceBuffer[DType.uint8]
    var mesh_count: Int
    var materials_buf: DeviceBuffer[DType.uint8]
    var material_count: Int
    # Keep all per-mesh device buffers alive
    var points_bufs: List[DeviceBuffer[DType.uint8]]
    var faceIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var vertexIndices_bufs: List[DeviceBuffer[DType.uint8]]

@export
fn mojo_gpu_upload_scene(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    bvh2NodesCount: Int64,
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    primIdsCount: Int64,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    meshCount: Int64,
    # Per-mesh sizes needed to upload vertex/index data
    meshPointsCounts: UnsafePointer[Int64, MutAnyOrigin],       # number of Float32 elements in each mesh's points array
    meshFaceIndicesCounts: UnsafePointer[Int64, MutAnyOrigin],  # number of Int64 elements
    meshVertexIndicesCounts: UnsafePointer[Int64, MutAnyOrigin], # number of Int64 elements
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    materialCount: Int64,
) -> UnsafePointer[GpuSceneHandle, MutAnyOrigin]:
    comptime if has_accelerator():
        try:
            var ctx = DeviceContext()
    
            # Check GPU memory
            var mem_info = ctx.get_memory_info()
            var free_bytes = mem_info[0]
            var total_bytes = mem_info[1]
    
            var bvh_bytes = Int(bvh2NodesCount) * 32  # sizeof(BVH2Node) = 32
            var prim_bytes = Int(primIdsCount) * 32    # sizeof(PrimId_C) = 32 (8+8+8+1+7 padding)
            var mesh_struct_bytes = Int(meshCount) * 24  # sizeof(TriangleMesh_C) = 3 pointers
            var material_struct_bytes = Int(materialCount) * 28 # sizeof(Material_C) = 1 int8 + 3 padding + 6 float32s
    
            # Estimate total mesh data
            var mesh_data_bytes = 0
            for i in range(Int(meshCount)):
                mesh_data_bytes += Int(meshPointsCounts[i]) * 4       # Float32
                mesh_data_bytes += Int(meshFaceIndicesCounts[i]) * 8  # Int64
                mesh_data_bytes += Int(meshVertexIndicesCounts[i]) * 8 # Int64
    
            var total_scene_bytes = bvh_bytes + prim_bytes + mesh_struct_bytes + mesh_data_bytes
            var free_mb = free_bytes // (1024 * 1024)
            var scene_mb = total_scene_bytes // (1024 * 1024)
    
            print("GPU: " + String(ctx.name()) + " — " + String(free_mb) + " MB free / " + String(total_bytes // (1024*1024)) + " MB total")
            print("GPU: Scene requires ~" + String(scene_mb) + " MB (" + String(bvh_bytes // (1024*1024)) + " MB BVH, " + String(mesh_data_bytes // (1024*1024)) + " MB mesh data)")
    
            if total_scene_bytes > Int(free_bytes):
                print("WARNING: Scene (" + String(scene_mb) + " MB) may exceed available GPU memory (" + String(free_mb) + " MB)!")
    
            # Upload BVH nodes
            var bvh_buf = ctx.enqueue_create_buffer[DType.uint8](bvh_bytes)
            with bvh_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = bvh2Nodes.bitcast[UInt8]()
                for i in range(bvh_bytes):
                    dst[i] = src[i]
    
            # Upload prim IDs
            var prim_buf = ctx.enqueue_create_buffer[DType.uint8](prim_bytes)
            with prim_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = primIds.bitcast[UInt8]()
                for i in range(prim_bytes):
                    dst[i] = src[i]
    
            # Upload per-mesh vertex/index data and build device-side mesh structs
            var points_bufs = List[DeviceBuffer[DType.uint8]]()
            var face_bufs = List[DeviceBuffer[DType.uint8]]()
            var vert_bufs = List[DeviceBuffer[DType.uint8]]()
    
            # We'll build mesh structs with device pointers
            var mesh_structs_host = alloc[TriangleMesh_C](Int(meshCount))
    
            for i in range(Int(meshCount)):
                var host_mesh = meshes[i]
    
                # Upload points
                var pts_count = Int(meshPointsCounts[i])
                var pts_bytes = pts_count * 4
                var pts_buf = ctx.enqueue_create_buffer[DType.uint8](pts_bytes)
                with pts_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.points.bitcast[UInt8]()
                    for j in range(pts_bytes):
                        dst[j] = src[j]
    
                # Upload face indices
                var fi_count = Int(meshFaceIndicesCounts[i])
                var fi_bytes = fi_count * 8
                var fi_buf = ctx.enqueue_create_buffer[DType.uint8](fi_bytes)
                with fi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.faceIndices.bitcast[UInt8]()
                    for j in range(fi_bytes):
                        dst[j] = src[j]
    
                # Upload vertex indices
                var vi_count = Int(meshVertexIndicesCounts[i])
                var vi_bytes = vi_count * 8
                var vi_buf = ctx.enqueue_create_buffer[DType.uint8](vi_bytes)
                with vi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.vertexIndices.bitcast[UInt8]()
                    for j in range(vi_bytes):
                        dst[j] = src[j]
    
                # Build mesh struct with device pointers
                mesh_structs_host[i] = TriangleMesh_C(
                    pts_buf.unsafe_ptr().bitcast[Float32](),
                    fi_buf.unsafe_ptr().bitcast[Int64](),
                    vi_buf.unsafe_ptr().bitcast[Int64](),
                )
    
                points_bufs.append(pts_buf^)
                face_bufs.append(fi_buf^)
                vert_bufs.append(vi_buf^)
    
            # Upload mesh struct array
            var meshes_buf = ctx.enqueue_create_buffer[DType.uint8](mesh_struct_bytes)
            with meshes_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = mesh_structs_host.bitcast[UInt8]()
                for j in range(mesh_struct_bytes):
                    dst[j] = src[j]
    
            mesh_structs_host.free()
    
            # Upload materials array
            var mat_bytes = Int(materialCount) * 28 # sizeof(Material_C) = 1 + 3 pad + 6 floats = 28
            var mat_buf = ctx.enqueue_create_buffer[DType.uint8](mat_bytes)
            if Int(materialCount) > 0:
                with mat_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = materials.bitcast[UInt8]()
                    for j in range(mat_bytes):
                        dst[j] = src[j]
    
            ctx.synchronize()
    
            # Allocate handle on heap
            var handle = alloc[GpuSceneHandle](1)
            handle.init_pointee_move(GpuSceneHandle(
                ctx=ctx^,
                bvh2Nodes_buf=bvh_buf^,
                primIds_buf=prim_buf^,
                meshes_buf=meshes_buf^,
                mesh_count=Int(meshCount),
                materials_buf=mat_buf^,
                material_count=Int(materialCount),
                points_bufs=points_bufs^,
                faceIndices_bufs=face_bufs^,
                vertexIndices_bufs=vert_bufs^,
            ))
    
            print("GPU: Scene uploaded successfully")
            return handle.bitcast[GpuSceneHandle]()
        except e:
            print("GPU: Failed to upload scene: " + String(e))
            return UnsafePointer[GpuSceneHandle, MutAnyOrigin]()
    else:
        return UnsafePointer[GpuSceneHandle, MutAnyOrigin]()


# GPU kernel function — one thread per ray
fn traverse_bvh2_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    rays: UnsafePointer[Ray_C, MutAnyOrigin],
    tMaxValues: UnsafePointer[Float32, MutAnyOrigin],
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = rays[tid]
    var tMax = tMaxValues[tid]
    var result_ptr = results + tid
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, ray, tMax, result_ptr)

@export
fn mojo_gpu_traverse_batch(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    rays: UnsafePointer[Ray_C, MutAnyOrigin],
    tMaxValues: UnsafePointer[Float32, MutAnyOrigin],
    count: Int64,
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
):
    if not handlePtr:
        return
    var handle = handlePtr

    var n = Int(count)
    if n == 0:
        return

    comptime if has_accelerator():
        try:
            # Upload rays to GPU
            var ray_bytes = n * 24  # sizeof(Ray_C) = 6 * 4 = 24
            var ray_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](ray_bytes)
            with ray_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = rays.bitcast[UInt8]()
                for i in range(ray_bytes):
                    dst[i] = src[i]
    
            # Upload tMax values
            var tmax_bytes = n * 4
            var tmax_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](tmax_bytes)
            with tmax_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = tMaxValues.bitcast[UInt8]()
                for i in range(tmax_bytes):
                    dst[i] = src[i]
    
            # Create output buffer
            var result_bytes = n * 48  # sizeof(Intersection_C) = PrimId(32) + f32*3(12) + i8*4(4) = 48
            var result_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](result_bytes)
    
            # Launch kernel
            comptime block_size = 256
            var grid_dim = ceildiv(n, block_size)
    
            handle[].ctx.enqueue_function[traverse_bvh2_gpu, traverse_bvh2_gpu](
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                ray_buf.unsafe_ptr().bitcast[Ray_C](),
                tmax_buf.unsafe_ptr().bitcast[Float32](),
                result_buf.unsafe_ptr().bitcast[Intersection_C](),
                n,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            
            handle[].ctx.synchronize()
    
            # Copy results back to host
            with result_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = results.bitcast[UInt8]()
                for i in range(result_bytes):
                    dst[i] = src[i]
        except e:
            print("GPU: Batch traversal failed: " + String(e))


@always_inline
fn shade_core(
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
    
    if mat.type == 2:
        path_ptr[].estimateR += path_ptr[].throughputR * mat.emissionR
        path_ptr[].estimateG += path_ptr[].throughputG * mat.emissionG
        path_ptr[].estimateB += path_ptr[].throughputB * mat.emissionB
        path_ptr[].active = 0
        return
        
    if mat.type == 1:
        # Construct Normal
        var mesh_idx: Int 
        var base_vidx: Int 
        if inter.primId.type == 0:
            mesh_idx = Int(inter.primId.id1)
            base_vidx = Int(inter.primId.id2)
        elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
            mesh_idx = Int(inter.primId.id2 >> 32)
            base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        else:
            path_ptr[].active = 0
            return

        var mesh = meshes[mesh_idx]
        var v0_idx = Int(mesh.vertexIndices[base_vidx])
        var v1_idx = Int(mesh.vertexIndices[base_vidx + 1])
        var v2_idx = Int(mesh.vertexIndices[base_vidx + 2])

        var p0 = SIMD[DType.float32, 3](mesh.points[v0_idx * 4], mesh.points[v0_idx * 4 + 1], mesh.points[v0_idx * 4 + 2])
        var p1 = SIMD[DType.float32, 3](mesh.points[v1_idx * 4], mesh.points[v1_idx * 4 + 1], mesh.points[v1_idx * 4 + 2])
        var p2 = SIMD[DType.float32, 3](mesh.points[v2_idx * 4], mesh.points[v2_idx * 4 + 1], mesh.points[v2_idx * 4 + 2])

        var edge1 = p1 - p0
        var edge2 = p2 - p0
        var normal = cross(edge1, edge2)
        var nlen = dot(normal, normal)
        if nlen > 0:
            normal = normal * (1.0 / sqrt(nlen))
            
        # Orient normal towards ray
        var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
        if dot(normal, ray_dir) > 0:
            normal = normal * -1.0
            
        # Cosine weighted hemisphere sampling
        var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state
        
        var r = sqrt(u1)
        var theta = 2.0 * Float32(3.14159265359) * u2
        var x = r * cos(theta)
        var y = r * sin(theta)
        var z2 = 1.0 - u1
        var z = sqrt(z2 if z2 > 0.0 else Float32(0.0))
        
        # Build tangent basis
        var sign = Float32(1.0) if normal[2] >= 0.0 else Float32(-1.0)
        var a = Float32(-1.0) / (sign + normal[2])
        var b = normal[0] * normal[1] * a
        var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
        var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > 0:
            dir = dir * (1.0 / sqrt(dlen))
            
        # Update Ray
        var org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ) + ray_dir * inter.tHit + normal * 0.0001
        path_ptr[].ray = Ray_C(org[0], org[1], org[2], dir[0], dir[1], dir[2])
        
        # Update Throughput (albedo)
        path_ptr[].throughputR *= mat.albedoR
        path_ptr[].throughputG *= mat.albedoG
        path_ptr[].throughputB *= mat.albedoB
    else:
        # Unknown material type — deactivate to prevent infinite loops
        path_ptr[].active = 0


# CPU-only shading with next-event estimation (shadow rays via any_hit_bvh2_core).
@always_inline
fn shade_core_cpu_nee(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
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

    # Emissive hit: add emission only if camera ray directly sees the light (bounce 0)
    if mat.type == 2:
        if path_ptr[].bounce == 0:
            path_ptr[].estimateR += path_ptr[].throughputR * mat.emissionR
            path_ptr[].estimateG += path_ptr[].throughputG * mat.emissionG
            path_ptr[].estimateB += path_ptr[].throughputB * mat.emissionB
        path_ptr[].active = 0
        return

    if mat.type == 3:
        shade_conductor(path_ptr, inter, meshes, mat)
        return

    if mat.type == 4:
        shade_dielectric(path_ptr, inter, meshes, mat)
        return

    if mat.type == 5:
        shade_coated_diffuse(path_ptr, inter, meshes, mat)
        return

    if mat.type == 6:
        shade_diffuse_transmission(path_ptr, inter, meshes, mat)
        return

    if mat.type != 1:
        path_ptr[].active = 0
        return

    # Resolve hit geometry
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0_idx = Int(mesh.vertexIndices[base_vidx])
    var v1_idx = Int(mesh.vertexIndices[base_vidx + 1])
    var v2_idx = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0_idx*4], mesh.points[v0_idx*4+1], mesh.points[v0_idx*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1_idx*4], mesh.points[v1_idx*4+1], mesh.points[v1_idx*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2_idx*4], mesh.points[v2_idx*4+1], mesh.points[v2_idx*4+2])
    var edge1 = p1 - p0
    var edge2 = p2 - p0
    var normal = cross(edge1, edge2)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    # Single PCG instance for all sampling in this shade step
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # ── Next-event estimation ──────────────────────────────────────────────────
    if areaLightCount > 0:
        var light_idx = Int(pcg.next_uint() % UInt32(areaLightCount))
        var al = areaLights[light_idx]

        var lmesh = meshes[Int(al.meshIdx)]
        var lb = Int(al.triBaseVidx)
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])

        var r1 = pcg.next_float()
        var r2 = pcg.next_float()
        var sqrt_r1 = sqrt(r1)
        var light_point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - r2)) + lp2 * (sqrt_r1 * r2)

        var ledge1 = lp1 - lp0
        var ledge2 = lp2 - lp0
        var lcross = cross(ledge1, ledge2)
        var lcross_len = dot(lcross, lcross)
        var light_area = Float32(0.5) * sqrt(lcross_len)
        var light_normal = lcross
        if lcross_len > Float32(0.0):
            light_normal = lcross * (Float32(1.0) / sqrt(lcross_len))

        var to_light = light_point - hit_point
        var dist_sq = dot(to_light, to_light)
        var dist = sqrt(dist_sq)

        if dist > Float32(0.0001) and light_area > Float32(0.0):
            var shadow_dir = to_light * (Float32(1.0) / dist)
            var cos_s = dot(normal, shadow_dir)
            var cos_l = -dot(light_normal, shadow_dir)
            if cos_s > Float32(0.0) and cos_l > Float32(0.0):
                var shadow_ray = Ray_C(hit_point[0], hit_point[1], hit_point[2],
                                      shadow_dir[0], shadow_dir[1], shadow_dir[2])
                if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, dist * Float32(0.9999)):
                    # pdf = 1 / (areaLightCount * light_area); G = cos_s * cos_l / dist_sq
                    var weight = cos_s * cos_l * light_area * Float32(areaLightCount) / dist_sq
                    path_ptr[].estimateR += path_ptr[].throughputR * mat.albedoR * al.emissionR * weight
                    path_ptr[].estimateG += path_ptr[].throughputG * mat.albedoG * al.emissionG * weight
                    path_ptr[].estimateB += path_ptr[].throughputB * mat.albedoB * al.emissionB * weight

    # ── Indirect: cosine-weighted hemisphere bounce ────────────────────────────
    var u1 = pcg.next_float()
    var u2 = pcg.next_float()

    var r = sqrt(u1)
    var theta = Float32(2.0) * Float32(3.14159265359) * u2
    var x = r * cos(theta)
    var y = r * sin(theta)
    var z2 = Float32(1.0) - u1
    var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))

    var sign = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
    var a = Float32(-1.0) / (sign + normal[2])
    var b = normal[0] * normal[1] * a
    var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
    var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])

    var dir = tangent * x + bitangent * y + normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))

    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])
    path_ptr[].throughputR *= mat.albedoR
    path_ptr[].throughputG *= mat.albedoG
    path_ptr[].throughputB *= mat.albedoB
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = Float32(0.2126) * path_ptr[].throughputR + Float32(0.7152) * path_ptr[].throughputG + Float32(0.0722) * path_ptr[].throughputB
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            var inv = Float32(1.0) / (Float32(1.0) - q)
            path_ptr[].throughputR *= inv
            path_ptr[].throughputG *= inv
            path_ptr[].throughputB *= inv

    path_ptr[].pcgState = pcg.state


# ── DiffuseTransmission branch — called from shade_core_cpu_nee ──────────────
# Stochastically reflects or transmits through the surface using the balance
# heuristic. albedo = reflectance, emission = transmittance (both * scale).
# Throughput is weighted by color / selection_probability so the estimator
# remains unbiased. No NEE — direct light comes via random walk.
@always_inline
fn shade_diffuse_transmission(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    # Orient normal toward incoming ray
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Balance heuristic: choose reflect vs transmit proportional to luminance
    var reflR = mat.albedoR;  var reflG = mat.albedoG;  var reflB = mat.albedoB
    var transR = mat.emissionR; var transG = mat.emissionG; var transB = mat.emissionB
    var pr = Float32(0.2126)*reflR  + Float32(0.7152)*reflG  + Float32(0.0722)*reflB
    var pt = Float32(0.2126)*transR + Float32(0.7152)*transG + Float32(0.0722)*transB
    var total = pr + pt
    if total <= Float32(0.0):
        path_ptr[].active = 0
        return

    var choose_reflect = pcg.next_float() < pr / total
    # Bounce normal: same side for reflection, opposite for transmission
    var bounce_normal = normal if choose_reflect else -normal

    # Cosine-weighted hemisphere around bounce_normal
    var u1 = pcg.next_float()
    var u2 = pcg.next_float()
    var r = sqrt(u1)
    var theta = Float32(2.0) * Float32(3.14159265359) * u2
    var x = r * cos(theta)
    var y = r * sin(theta)
    var z2 = Float32(1.0) - u1
    var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))

    var sign = Float32(1.0) if bounce_normal[2] >= Float32(0.0) else Float32(-1.0)
    var a = Float32(-1.0) / (sign + bounce_normal[2])
    var b = bounce_normal[0] * bounce_normal[1] * a
    var tangent  = SIMD[DType.float32, 3](Float32(1.0) + sign * bounce_normal[0] * bounce_normal[0] * a, sign * b, -sign * bounce_normal[0])
    var bitangent = SIMD[DType.float32, 3](b, sign + bounce_normal[1] * bounce_normal[1] * a, -bounce_normal[1])

    var dir = tangent * x + bitangent * y + bounce_normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))

    # Offset along bounce_normal to avoid self-intersection
    var hit_point = ray_org + ray_dir * inter.tHit + bounce_normal * Float32(0.0001)
    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])

    # Throughput weight = color / selection_probability = color * total / p_choice
    if choose_reflect:
        var w = total / pr
        path_ptr[].throughputR *= reflR * w
        path_ptr[].throughputG *= reflG * w
        path_ptr[].throughputB *= reflB * w
    else:
        var w = total / pt
        path_ptr[].throughputR *= transR * w
        path_ptr[].throughputG *= transG * w
        path_ptr[].throughputB *= transB * w

    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = Float32(0.2126) * path_ptr[].throughputR + Float32(0.7152) * path_ptr[].throughputG + Float32(0.0722) * path_ptr[].throughputB
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            var inv = Float32(1.0) / (Float32(1.0) - q)
            path_ptr[].throughputR *= inv
            path_ptr[].throughputG *= inv
            path_ptr[].throughputB *= inv

    path_ptr[].pcgState = pcg.state


# ── CoatedDiffuse (plastic) branch — called from shade_core_cpu_nee ─────────
# Fresnel coating over diffuse substrate. IOR stored in mat.emissionR.
# Stochastically routes to specular reflection (Fresnel weight) or cosine
# hemisphere diffuse bounce (throughput *= albedo).
@always_inline
fn shade_coated_diffuse(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # Schlick Fresnel for the dielectric coating (IOR stored in emissionR)
    var ior = mat.emissionR
    var cos_i = -dot(ray_dir, normal)
    var r0 = (Float32(1.0) - ior) / (Float32(1.0) + ior)
    r0 = r0 * r0
    var one_minus = Float32(1.0) - cos_i
    var one_minus2 = one_minus * one_minus
    var fresnel = r0 + (Float32(1.0) - r0) * one_minus2 * one_minus2 * one_minus

    if pcg.next_float() < fresnel:
        # Specular reflection from coating — throughput unchanged
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    else:
        # Diffuse bounce through coating
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        var r = sqrt(u1)
        var theta = Float32(2.0) * Float32(3.14159265359) * u2
        var x = r * cos(theta)
        var y = r * sin(theta)
        var z2 = Float32(1.0) - u1
        var z = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))

        var sign = Float32(1.0) if normal[2] >= Float32(0.0) else Float32(-1.0)
        var a = Float32(-1.0) / (sign + normal[2])
        var b = normal[0] * normal[1] * a
        var tangent = SIMD[DType.float32, 3](Float32(1.0) + sign * normal[0] * normal[0] * a, sign * b, -sign * normal[0])
        var bitangent = SIMD[DType.float32, 3](b, sign + normal[1] * normal[1] * a, -normal[1])

        var dir = tangent * x + bitangent * y + normal * z
        var dlen = dot(dir, dir)
        if dlen > Float32(0.0):
            dir = dir * (Float32(1.0) / sqrt(dlen))

        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], dir[0], dir[1], dir[2])
        path_ptr[].throughputR *= mat.albedoR
        path_ptr[].throughputG *= mat.albedoG
        path_ptr[].throughputB *= mat.albedoB

    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    if path_ptr[].bounce > 1:
        var lum = Float32(0.2126) * path_ptr[].throughputR + Float32(0.7152) * path_ptr[].throughputG + Float32(0.0722) * path_ptr[].throughputB
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            var inv = Float32(1.0) / (Float32(1.0) - q)
            path_ptr[].throughputR *= inv
            path_ptr[].throughputG *= inv
            path_ptr[].throughputB *= inv

    path_ptr[].pcgState = pcg.state


# ── Dielectric (glass) branch — called from shade_core_cpu_nee ──────────────
@always_inline
fn shade_dielectric(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var geom_normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(geom_normal, geom_normal)
    if nlen > Float32(0.0):
        geom_normal = geom_normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)

    var ior = mat.albedoR
    var entering = dot(ray_dir, geom_normal) < Float32(0.0)
    var normal = geom_normal if entering else -geom_normal
    var eta = (Float32(1.0) / ior) if entering else ior

    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)

    # Schlick Fresnel
    var r0 = (Float32(1.0) - ior) / (Float32(1.0) + ior)
    r0 = r0 * r0
    var one_minus = Float32(1.0) - cos_i
    var one_minus2 = one_minus * one_minus
    var fresnel = r0 + (Float32(1.0) - r0) * one_minus2 * one_minus2 * one_minus

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if tir or pcg.next_float() < fresnel:
        # Reflect: r = d + 2*cos_i*n  (derived from r = d - 2(d·n)n, d·n = -cos_i)
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    else:
        # Refract: t = eta*d + (eta*cos_i - sqrt(1 - sin2_t))*n
        var cos_t = sqrt(Float32(1.0) - sin2_t)
        var refr = ray_dir * eta + normal * (eta * cos_i - cos_t)
        var rlen = dot(refr, refr)
        if rlen > Float32(0.0):
            refr = refr * (Float32(1.0) / sqrt(rlen))
        var hit_point = ray_org + ray_dir * inter.tHit - normal * Float32(0.0001)
        path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refr[0], refr[1], refr[2])

    path_ptr[].bounce += 1

    # Russian roulette after first bounce (throughput unchanged for ideal glass)
    if path_ptr[].bounce > 1:
        var lum = Float32(0.2126) * path_ptr[].throughputR + Float32(0.7152) * path_ptr[].throughputG + Float32(0.0722) * path_ptr[].throughputB
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            var inv = Float32(1.0) / (Float32(1.0) - q)
            path_ptr[].throughputR *= inv
            path_ptr[].throughputG *= inv
            path_ptr[].throughputB *= inv

    path_ptr[].pcgState = pcg.state


# ── Conductor (perfect mirror) branch — called from shade_core_cpu_nee ──────
@always_inline
fn shade_conductor(
    path_ptr: UnsafePointer[PathState_C, MutAnyOrigin],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
):
    var mesh_idx: Int
    var base_vidx: Int
    if inter.primId.type == 0:
        mesh_idx = Int(inter.primId.id1)
        base_vidx = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mesh_idx = Int(inter.primId.id2 >> 32)
        base_vidx = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        path_ptr[].active = 0
        return

    var mesh = meshes[mesh_idx]
    var v0 = Int(mesh.vertexIndices[base_vidx])
    var v1 = Int(mesh.vertexIndices[base_vidx + 1])
    var v2 = Int(mesh.vertexIndices[base_vidx + 2])
    var p0 = SIMD[DType.float32, 3](mesh.points[v0*4], mesh.points[v0*4+1], mesh.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](mesh.points[v1*4], mesh.points[v1*4+1], mesh.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](mesh.points[v2*4], mesh.points[v2*4+1], mesh.points[v2*4+2])

    var normal = cross(p1 - p0, p2 - p0)
    var nlen = dot(normal, normal)
    if nlen > Float32(0.0):
        normal = normal * (Float32(1.0) / sqrt(nlen))

    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.dirX, path_ptr[].ray.dirY, path_ptr[].ray.dirZ)
    if dot(normal, ray_dir) > Float32(0.0):
        normal = -normal

    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.orgX, path_ptr[].ray.orgY, path_ptr[].ray.orgZ)
    var hit_point = ray_org + ray_dir * inter.tHit + normal * Float32(0.0001)

    # Perfect specular reflection: r = d - 2(d·n)n
    var refl = ray_dir - normal * (Float32(2.0) * dot(ray_dir, normal))
    var rlen = dot(refl, refl)
    if rlen > Float32(0.0):
        refl = refl * (Float32(1.0) / sqrt(rlen))

    path_ptr[].ray = Ray_C(hit_point[0], hit_point[1], hit_point[2], refl[0], refl[1], refl[2])
    path_ptr[].throughputR *= mat.albedoR
    path_ptr[].throughputG *= mat.albedoG
    path_ptr[].throughputB *= mat.albedoB
    path_ptr[].bounce += 1

    # Russian roulette after first bounce
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    if path_ptr[].bounce > 1:
        var lum = Float32(0.2126) * path_ptr[].throughputR + Float32(0.7152) * path_ptr[].throughputG + Float32(0.0722) * path_ptr[].throughputB
        var q = Float32(1.0) - (lum if lum < Float32(0.95) else Float32(0.95))
        if pcg.next_float() < q:
            path_ptr[].active = 0
        else:
            var inv = Float32(1.0) / (Float32(1.0) - q)
            path_ptr[].throughputR *= inv
            path_ptr[].throughputG *= inv
            path_ptr[].throughputB *= inv
    path_ptr[].pcgState = pcg.state


fn shade_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shade_core(paths, intersections, meshes, materials, tid)

@export
fn mojo_gpu_shade_batch(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    count: Int64,
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin]
):
    if not handlePtr:
        return
    var handle = handlePtr
    var n = Int(count)
    if n == 0:
        return

    comptime if has_accelerator():
        try:
            # Create mapped unmanaged device buffers based on exact sizes
            var path_bytes = n * 88 # sizeof(PathState_C) = 88
            var inter_bytes = n * 48 # sizeof(Intersection_C) = 48
            
            var path_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](path_bytes)
            with path_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = paths.bitcast[UInt8]()
                for i in range(path_bytes):
                    dst[i] = src[i]
                    
            var inter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](inter_bytes)
            with inter_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = intersections.bitcast[UInt8]()
                for i in range(inter_bytes):
                    dst[i] = src[i]
                    
            # Launch shape kernel
            comptime block_size = 256
            var grid_dim = ceildiv(n, block_size)
    
            handle[].ctx.enqueue_function[shade_gpu, shade_gpu](
                path_buf.unsafe_ptr().bitcast[PathState_C](),
                inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                n,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            
            handle[].ctx.synchronize()
            
            # Transfer path back (they were updated in-place on the device)
            with path_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = paths.bitcast[UInt8]()
                for i in range(path_bytes):
                    dst[i] = src[i]
                    
        except e:
            print("GPU: Batch shading failed: " + String(e))

@export
def mojo_render_paths(
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    count: Int64,
    maxDepth: Int32,
):
    var scene = scenePtr[0]
    var n = Int(count)
    var maxD = Int(maxDepth)

    var intersections = alloc[Intersection_C](n)

    for bounce in range(maxD + 1):
        var anyActive = False
        for i in range(n):
            if paths[i].active != 0:
                anyActive = True
                break
        if not anyActive:
            break

        for i in range(n):
            if paths[i].active == 0:
                continue
            traverse_bvh2_core(
                scene.bvh2Nodes, scene.primIds, scene.meshes,
                paths[i].ray, Float32(1.0e38), intersections + i,
            )

        for i in range(n):
            if paths[i].active == 0:
                continue
            shade_core_cpu_nee(paths, intersections, scene.bvh2Nodes, scene.primIds,
                               scene.meshes, scene.materials,
                               scene.areaLights, Int(scene.areaLightCount), i)

    intersections.free()


# ── ZSobolSampler + GaussianFilter ported from Swift ─────────────────────────

@fieldwise_init
struct TileSamplerParams_C(TrivialRegisterPassable):
    var sobolMatrices: UnsafePointer[UInt32, MutAnyOrigin]
    var rngSeed: UInt64
    var sobolSeed: Int32
    var log2SamplesPerPixel: Int32
    var nBase4Digits: Int32
    var samplesPerPixel: Int32
    var filterSigma: Float32
    var filterSupportX: Float32
    var filterSupportY: Float32
    var filterNormX: Float32
    var filterNormY: Float32
    var filterWeight: Float32

@always_inline
fn reverse_bits32(v_in: UInt32) -> UInt32:
    var v = v_in
    v = ((v >> 1) & UInt32(0x55555555)) | ((v & UInt32(0x55555555)) << 1)
    v = ((v >> 2) & UInt32(0x33333333)) | ((v & UInt32(0x33333333)) << 2)
    v = ((v >> 4) & UInt32(0x0f0f0f0f)) | ((v & UInt32(0x0f0f0f0f)) << 4)
    v = ((v >> 8) & UInt32(0x00ff00ff)) | ((v & UInt32(0x00ff00ff)) << 8)
    return (v >> 16) | (v << 16)

@always_inline
fn fast_owen_scramble(value_in: UInt32, seed: UInt32) -> UInt32:
    var v = reverse_bits32(value_in)
    v ^= v * UInt32(0x3d20adea)
    v += seed
    v *= (seed >> 16) | UInt32(1)
    v ^= v * UInt32(0x05526c56)
    v ^= v * UInt32(0x53a22864)
    return reverse_bits32(v)

@always_inline
fn mix_bits_u64(v: UInt64) -> UInt32:
    var v32 = UInt32(v & UInt64(0xFFFFFFFF))
    v32 ^= UInt32(v >> 32)
    v32 ^= v32 >> 16
    v32 *= UInt32(0x85ebca77)
    v32 ^= v32 >> 13
    v32 *= UInt32(0xc2b2ae35)
    v32 ^= v32 >> 16
    return v32

@always_inline
fn encode_morton2(x: UInt32, y: UInt32) -> UInt64:
    var x64 = UInt64(x)
    var y64 = UInt64(y)
    x64 = (x64 | (x64 << 16)) & UInt64(0x0000FFFF0000FFFF)
    x64 = (x64 | (x64 << 8))  & UInt64(0x00FF00FF00FF00FF)
    x64 = (x64 | (x64 << 4))  & UInt64(0x0F0F0F0F0F0F0F0F)
    x64 = (x64 | (x64 << 2))  & UInt64(0x3333333333333333)
    x64 = (x64 | (x64 << 1))  & UInt64(0x5555555555555555)
    y64 = (y64 | (y64 << 16)) & UInt64(0x0000FFFF0000FFFF)
    y64 = (y64 | (y64 << 8))  & UInt64(0x00FF00FF00FF00FF)
    y64 = (y64 | (y64 << 4))  & UInt64(0x0F0F0F0F0F0F0F0F)
    y64 = (y64 | (y64 << 2))  & UInt64(0x3333333333333333)
    y64 = (y64 | (y64 << 1))  & UInt64(0x5555555555555555)
    return x64 | (y64 << 1)

# Compact permutation encoding: each of 24 permutations of {0,1,2,3} stored in one UInt8.
# Encoding: enc = (d0<<6)|(d1<<4)|(d2<<2)|d3. Lookup: (enc >> (2*(3-digit))) & 3.
# Values computed from ZSobolSampler.permutations in Swift.
@always_inline
fn sobol_perm_lookup(p_idx: Int, digit: Int) -> Int:
    var enc = InlineArray[UInt8, 24](fill=UInt8(0))
    enc[ 0]=27; enc[ 1]=30; enc[ 2]=39; enc[ 3]=45; enc[ 4]=57; enc[ 5]=54
    enc[ 6]=75; enc[ 7]=78; enc[ 8]=99; enc[ 9]=108; enc[10]=120; enc[11]=114
    enc[12]=147; enc[13]=156; enc[14]=135; enc[15]=141; enc[16]=177; enc[17]=180
    enc[18]=216; enc[19]=210; enc[20]=228; enc[21]=225; enc[22]=201; enc[23]=198
    return Int((Int(enc[p_idx]) >> (2 * (3 - digit))) & 3)

@always_inline
fn sobol_get_sample_index(
    morton_idx: UInt64, dim: Int, log2spp: Int, n_base4: Int,
) -> UInt64:
    var sample_index: UInt64 = 0
    var pow2_samples = (log2spp & 1) == 1
    var last_digit = 1 if pow2_samples else 0
    var digit_index = n_base4 - 1
    while digit_index >= last_digit:
        var digit_shift = 2 * digit_index - (1 if pow2_samples else 0)
        var digit = Int((morton_idx >> UInt64(digit_shift)) & UInt64(3))
        var higher_digits = morton_idx >> UInt64(digit_shift + 2)
        var hash_val = mix_bits_u64(higher_digits ^ (UInt64(0x55555555) * UInt64(dim)))
        var p_idx = Int((hash_val >> 24) % UInt32(24))
        digit = sobol_perm_lookup(p_idx, digit)
        sample_index |= UInt64(digit) << UInt64(digit_shift)
        digit_index -= 1
    if pow2_samples:
        var digit = Int(morton_idx & UInt64(1))
        var hash_val = mix_bits_u64((morton_idx >> 1) ^ (UInt64(0x55555555) * UInt64(dim)))
        digit ^= Int(hash_val & UInt32(1))
        sample_index |= UInt64(digit)
    return sample_index

@always_inline
fn sobol_sample(
    index: Int, dim: Int, seed: UInt32,
    matrices: UnsafePointer[UInt32, MutAnyOrigin],
) -> Float32:
    var acc: UInt32 = 0
    var cur = index
    var base = dim * 52
    for bit in range(52):
        if cur & 1 != 0:
            acc ^= matrices[base + bit]
        cur >>= 1
        if cur == 0:
            break
    var scrambled = fast_owen_scramble(acc, seed)
    return min(Float32(scrambled) * Float32(2.32830643653869628906e-10), Float32(0.9999999))

# Polynomial erfinv — no Newton refinement, sufficient accuracy for filter sampling.
# Matches GaussianFilter.erfinv in Swift (same polynomial coefficients).
@always_inline
fn gaussian_erfinv(y: Float32) -> Float32:
    var abs_y = y if y >= Float32(0.0) else -y
    if abs_y <= Float32(0.7):
        var z = y * y
        var num = Float32(0.886226899) + z * (Float32(-1.645349621) + z * (Float32(0.914624893) + z * Float32(-0.140543331)))
        var den = Float32(1.0) + z * (Float32(-2.118377725) + z * (Float32(1.442710462) + z * (Float32(-0.329097515) + z * Float32(0.012229801))))
        return y * num / den
    elif abs_y < Float32(1.0):
        var z = sqrt(-log((Float32(1.0) - abs_y) / Float32(2.0)))
        var num = Float32(-1.970840454) + z * (Float32(-1.624906493) + z * (Float32(3.429567803) + z * Float32(1.641345311)))
        var den = Float32(1.0) + z * (Float32(3.543889200) + z * Float32(1.637067800))
        var sign_y = Float32(1.0) if y >= Float32(0.0) else Float32(-1.0)
        return sign_y * num / den
    else:
        return Float32(3.4e38) if y > Float32(0.0) else Float32(-3.4e38)

# Importance-sample a 1D Gaussian filter.
# norm = 0.5*(1+erf(radius/(sigma*sqrt(2)))) — pre-computed in Swift.
@always_inline
fn gaussian_sample_1d(u: Float32, norm: Float32, sigma: Float32, radius: Float32) -> Float32:
    var u_s = (Float32(1.0) - norm) + u * (Float32(2.0) * norm - Float32(1.0))
    var x = sigma * sqrt(Float32(2.0)) * gaussian_erfinv(Float32(2.0) * u_s - Float32(1.0))
    return max(-radius, min(radius, x))

# erf via Abramowitz & Stegun 7.1.26 (max error ≤ 1.5e-7).
fn _erf(x: Float32) -> Float32:
    var sign = Float32(1) if x >= Float32(0) else Float32(-1)
    var ax = x if x >= Float32(0) else -x
    var t = Float32(1) / (Float32(1) + Float32(0.3275911) * ax)
    var poly = ((((Float32(1.061405429) * t
                - Float32(1.453152027)) * t
               + Float32(1.421413741)) * t
              - Float32(0.284496736)) * t
             + Float32(0.254829592)) * t
    return sign * (Float32(1) - poly * exp(-ax * ax))


# Gaussian filter normalization for mojo_render_tile_v2 (step 14).
# Returns 0.5*(1+erf(support/(sigma*sqrt(2)))), matching GaussianFilter.mojoParams().
@export
fn mojo_gaussian_norm(support: Float32, sigma: Float32) -> Float32:
    var x = support / (sigma * sqrt(Float32(2)))
    return Float32(0.5) * (Float32(1) + _erf(x))


# Hash pixel + sample index into a unique PCG (state, inc) pair.
@always_inline
fn derive_pcg_seeds(px: Int32, py: Int32, si: Int32, seed: UInt64) -> Tuple[UInt64, UInt64]:
    var h = UInt64(px) * UInt64(2654435761) ^ UInt64(py) * UInt64(1664525) ^ UInt64(si) * UInt64(22695477) ^ seed
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27; h *= UInt64(0x94d049bb133111eb)
    h ^= h >> 31
    var state = h
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27
    return (state, h | UInt64(1))

@export
def mojo_render_tile_v2(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld: UnsafePointer[Float32, MutAnyOrigin],
    tileMinX: Int32, tileMinY: Int32, tileMaxX: Int32, tileMaxY: Int32,
    samplerParamsPtr: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    resultsPtr: UnsafePointer[TileResult_C, MutAnyOrigin],
    maxDepth: Int32,
):
    var sp = samplerParamsPtr[0]
    var scene = scenePtr[0]
    var maxD = Int(maxDepth)
    var orgX = cameraToWorld[12]; var orgY = cameraToWorld[13]; var orgZ = cameraToWorld[14]
    var tileW = Int(tileMaxX - tileMinX)
    var tileH = Int(tileMaxY - tileMinY)
    var spp = Int(sp.samplesPerPixel)
    var n = tileW * tileH * spp
    var log2spp = Int(sp.log2SamplesPerPixel)
    var n_base4 = Int(sp.nBase4Digits)
    var matrices = sp.sobolMatrices

    # Seed hashes for Sobol dims 0 and 1 (matching ZSobolSampler.getSampleHash at dimension=0)
    # Swift: bits = UInt64(mixBits(UInt64(0) ^ UInt64(seed))) → lower32=sampleHash0, upper32=0
    var hash_bits = UInt64(mix_bits_u64(UInt64(0) ^ UInt64(sp.sobolSeed)))
    var seed_dim0 = UInt32(hash_bits & UInt64(0xFFFFFFFF))
    var seed_dim1 = UInt32(0)   # upper 32 bits of gonzales mixBits result = 0

    var paths = alloc[PathState_C](n)
    var intersections = alloc[Intersection_C](n)

    # Generate primary rays from Sobol film samples
    var idx = 0
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            var morton_base = encode_morton2(UInt32(px), UInt32(py)) << UInt64(log2spp)
            for si in range(spp):
                var morton_idx = morton_base | UInt64(si)
                var sobol_idx = sobol_get_sample_index(morton_idx, 0, log2spp, n_base4)
                var u0 = sobol_sample(Int(sobol_idx), 0, seed_dim0, matrices)
                var u1 = sobol_sample(Int(sobol_idx), 1, seed_dim1, matrices)
                var deltaX = gaussian_sample_1d(u0, sp.filterNormX, sp.filterSigma, sp.filterSupportX)
                var deltaY = gaussian_sample_1d(u1, sp.filterNormY, sp.filterSigma, sp.filterSupportY)
                var filmX = Float32(px) + Float32(0.5) + deltaX
                var filmY = Float32(py) + Float32(0.5) + deltaY

                # rasterToCamera transform (column-major)
                var cx = rasterToCamera[0]*filmX + rasterToCamera[4]*filmY + rasterToCamera[12]
                var cy = rasterToCamera[1]*filmX + rasterToCamera[5]*filmY + rasterToCamera[13]
                var cz = rasterToCamera[2]*filmX + rasterToCamera[6]*filmY + rasterToCamera[14]
                var cw = rasterToCamera[3]*filmX + rasterToCamera[7]*filmY + rasterToCamera[15]
                if cw != Float32(0.0) and cw != Float32(1.0):
                    cx /= cw; cy /= cw; cz /= cw
                var camDir = SIMD[DType.float32, 3](cx, cy, cz)
                var camLen = dot(camDir, camDir)
                if camLen > Float32(0.0):
                    camDir = camDir * (Float32(1.0) / sqrt(camLen))

                # cameraToWorld rotation (3×3)
                var dx = cameraToWorld[0]*camDir[0] + cameraToWorld[4]*camDir[1] + cameraToWorld[8]*camDir[2]
                var dy = cameraToWorld[1]*camDir[0] + cameraToWorld[5]*camDir[1] + cameraToWorld[9]*camDir[2]
                var dz = cameraToWorld[2]*camDir[0] + cameraToWorld[6]*camDir[1] + cameraToWorld[10]*camDir[2]
                var worldDir = SIMD[DType.float32, 3](dx, dy, dz)
                var dirLen = dot(worldDir, worldDir)
                if dirLen > Float32(0.0):
                    worldDir = worldDir * (Float32(1.0) / sqrt(dirLen))

                var (pcg_state, pcg_inc) = derive_pcg_seeds(px, py, Int32(si), sp.rngSeed)
                paths[idx] = PathState_C(
                    Ray_C(orgX, orgY, orgZ, worldDir[0], worldDir[1], worldDir[2]),
                    Float32(1.0), Float32(1.0), Float32(1.0),
                    Float32(0.0), Float32(0.0), Float32(0.0),
                    Float32(0.0), Float32(0.0), Float32(0.0),
                    Int32(0), pcg_state, pcg_inc,
                    Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0),
                )
                idx += 1

    # Multi-bounce path trace (identical to mojo_render_tile)
    for _ in range(maxD + 1):
        var anyActive = False
        for i in range(n):
            if paths[i].active != 0:
                anyActive = True
                break
        if not anyActive:
            break
        for i in range(n):
            if paths[i].active == 0:
                continue
            traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes,
                               paths[i].ray, Float32(1.0e38), intersections + i)
        for i in range(n):
            if paths[i].active == 0:
                continue
            shade_core_cpu_nee(paths, intersections, scene.bvh2Nodes, scene.primIds,
                               scene.meshes, scene.materials,
                               scene.areaLights, Int(scene.areaLightCount), i)

    # Accumulate the spp samples per pixel and emit one result per pixel.
    # Matches the Swift Film accumulation: pixel = Σ light / Σ weight.
    idx = 0
    var out = 0
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            var sumLR = Float32(0.0); var sumLG = Float32(0.0); var sumLB = Float32(0.0)
            var sumAR = Float32(0.0); var sumAG = Float32(0.0); var sumAB = Float32(0.0)
            var sumW = Float32(0.0)
            for _ in range(spp):
                sumLR += paths[idx].estimateR
                sumLG += paths[idx].estimateG
                sumLB += paths[idx].estimateB
                sumAR += paths[idx].albedoR
                sumAG += paths[idx].albedoG
                sumAB += paths[idx].albedoB
                sumW += sp.filterWeight
                idx += 1
            resultsPtr[out] = TileResult_C(
                sumLR, sumLG, sumLB, sumAR, sumAG, sumAB, sumW, px, py,
            )
            out += 1

    intersections.free()
    paths.free()


# ── BVH2 Construction (SAH) ───────────────────────────────────────────
# Builds a depth-first BVH2 node array from per-primitive AABBs, matching
# the Swift BoundingHierarchyBuilder layout: left child is at index+1, the
# right-child index is stored in .offset, leaves carry count>0 / offset=start.
# The working arrays widx/wmin/wmax are reordered in place; widx ends up
# holding the primitive permutation in leaf order.

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
        out_nodes[my] = BVH2Node(bminx, bminy, bminz, bmaxx, bmaxy, bmaxz,
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
        out_nodes[my] = BVH2Node(bminx, bminy, bminz, bmaxx, bmaxy, bmaxz,
                                 Int32(start), Int32(count))
        return Int32(my)

    # Interior node: left child is the next reserved slot (my+1).
    _ = build_bvh2_node(widx, wmin, wmax, start, mid,
                        out_nodes, node_count, prims_per_node)
    var right = build_bvh2_node(widx, wmin, wmax, mid, end,
                                out_nodes, node_count, prims_per_node)
    out_nodes[my] = BVH2Node(bminx, bminy, bminz, bmaxx, bmaxy, bmaxz,
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


@export
def mojo_render_tile(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld: UnsafePointer[Float32, MutAnyOrigin],
    samplesPtr: UnsafePointer[PixelSample_C, MutAnyOrigin],
    count: Int64,
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    resultsPtr: UnsafePointer[TileResult_C, MutAnyOrigin],
    maxDepth: Int32,
):
    var scene = scenePtr[0]
    var n = Int(count)
    var maxD = Int(maxDepth)

    # World-space camera origin = translation column of cameraToWorld (col 3)
    var orgX = cameraToWorld[12]
    var orgY = cameraToWorld[13]
    var orgZ = cameraToWorld[14]

    var paths = alloc[PathState_C](n)
    var intersections = alloc[Intersection_C](n)

    # Generate primary rays from film samples
    for i in range(n):
        var s = samplesPtr[i]
        var fX = s.filmX
        var fY = s.filmY

        # rasterToCamera * (fX, fY, 0, 1) — column-major: M[r,c] = flat[c*4+r]
        var cx = rasterToCamera[0]*fX + rasterToCamera[4]*fY + rasterToCamera[12]
        var cy = rasterToCamera[1]*fX + rasterToCamera[5]*fY + rasterToCamera[13]
        var cz = rasterToCamera[2]*fX + rasterToCamera[6]*fY + rasterToCamera[14]
        var cw = rasterToCamera[3]*fX + rasterToCamera[7]*fY + rasterToCamera[15]
        if cw != Float32(0.0) and cw != Float32(1.0):
            cx = cx / cw; cy = cy / cw; cz = cz / cw

        # Normalize camera-space direction
        var camDir = SIMD[DType.float32, 3](cx, cy, cz)
        var camLen = dot(camDir, camDir)
        if camLen > Float32(0.0):
            camDir = camDir * (Float32(1.0) / sqrt(camLen))

        # cameraToWorld * direction (3x3 part only, no translation)
        var dx = cameraToWorld[0]*camDir[0] + cameraToWorld[4]*camDir[1] + cameraToWorld[8]*camDir[2]
        var dy = cameraToWorld[1]*camDir[0] + cameraToWorld[5]*camDir[1] + cameraToWorld[9]*camDir[2]
        var dz = cameraToWorld[2]*camDir[0] + cameraToWorld[6]*camDir[1] + cameraToWorld[10]*camDir[2]
        var worldDir = SIMD[DType.float32, 3](dx, dy, dz)
        var dirLen = dot(worldDir, worldDir)
        if dirLen > Float32(0.0):
            worldDir = worldDir * (Float32(1.0) / sqrt(dirLen))

        paths[i] = PathState_C(
            Ray_C(orgX, orgY, orgZ, worldDir[0], worldDir[1], worldDir[2]),
            Float32(1.0), Float32(1.0), Float32(1.0),
            Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(0.0), Float32(0.0), Float32(0.0),
            Int32(0),
            s.pcgState, s.pcgInc,
            Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0),
        )

    # Multi-bounce path trace
    for bounce in range(maxD + 1):
        var anyActive = False
        for i in range(n):
            if paths[i].active != 0:
                anyActive = True
                break
        if not anyActive:
            break

        for i in range(n):
            if paths[i].active == 0:
                continue
            traverse_bvh2_core(
                scene.bvh2Nodes, scene.primIds, scene.meshes,
                paths[i].ray, Float32(1.0e38), intersections + i,
            )

        for i in range(n):
            if paths[i].active == 0:
                continue
            shade_core_cpu_nee(paths, intersections, scene.bvh2Nodes, scene.primIds,
                               scene.meshes, scene.materials,
                               scene.areaLights, Int(scene.areaLightCount), i)

    # Write results for film accumulation
    for i in range(n):
        var s = samplesPtr[i]
        resultsPtr[i] = TileResult_C(
            paths[i].estimateR, paths[i].estimateG, paths[i].estimateB,
            paths[i].albedoR, paths[i].albedoG, paths[i].albedoB,
            s.filterWeight, s.pixelX, s.pixelY,
        )

    intersections.free()
    paths.free()


# ── pbrt Lexer Helpers ────────────────────────────────────────────────
# Tile scheduling (step 15). Renders the full image by iterating over
# tiles sequentially; each tile is rendered by the existing mojo_render_tile_v2
# subroutine. CPU parallelism will be added in Step 19 when Mojo owns main().
# Results are written to a flat buffer: results[py*resX + px] for each pixel.

@export
def mojo_render_all_tiles(
    raster_to_camera: UnsafePointer[Float32, MutAnyOrigin],
    camera_to_world: UnsafePointer[Float32, MutAnyOrigin],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    tile_w: Int32, tile_h: Int32,
    sampler_params: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scene: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    results: UnsafePointer[TileResult_C, MutAnyOrigin],
    max_depth: Int32,
):
    var res_x = Int(max_x - min_x)
    var tw = Int(tile_w)
    var th = Int(tile_h)
    var max_tile_pixels = tw * th
    var tile_buf = alloc[TileResult_C](max_tile_pixels)

    var ty = Int(min_y)
    while ty < Int(max_y):
        var tx = Int(min_x)
        while tx < Int(max_x):
            var tx_max = Int32(min(tx + tw, Int(max_x)))
            var ty_max = Int32(min(ty + th, Int(max_y)))
            var tw_actual = Int(tx_max) - tx
            var th_actual = Int(ty_max) - ty
            mojo_render_tile_v2(
                raster_to_camera, camera_to_world,
                Int32(tx), Int32(ty), tx_max, ty_max,
                sampler_params, scene, tile_buf, max_depth)
            # Copy tile results into global buffer using absolute pixel coords.
            for iy in range(th_actual):
                for ix in range(tw_actual):
                    var src = iy * tw_actual + ix
                    var dst = (ty + iy - Int(min_y)) * res_x + (tx + ix - Int(min_x))
                    results[dst] = tile_buf[src]
            tx += tw
        ty += th

    tile_buf.free()


# Normalize TileResult_C[] → per-pixel float RGB arrays.
# beauty_out and albedo_out must each hold count*3 floats (R,G,B interleaved).
# Applies iso/100 scaling and optional maxComponentValue clamping to beauty.
@export
fn mojo_normalize_film(
    results: UnsafePointer[TileResult_C, MutAnyOrigin],
    count: Int32,
    iso: Float32,
    max_component_value: Float32,
    beauty_out: UnsafePointer[Float32, MutAnyOrigin],
    albedo_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var scale = iso / Float32(100)
    for i in range(Int(count)):
        var r = results[i]
        var w = r.filterWeight
        if w == Float32(0):
            beauty_out[i * 3 + 0] = Float32(0)
            beauty_out[i * 3 + 1] = Float32(0)
            beauty_out[i * 3 + 2] = Float32(0)
            albedo_out[i * 3 + 0] = Float32(0)
            albedo_out[i * 3 + 1] = Float32(0)
            albedo_out[i * 3 + 2] = Float32(0)
            continue
        var br = r.estimateR / w * scale
        var bg = r.estimateG / w * scale
        var bb = r.estimateB / w * scale
        if max_component_value > Float32(0):
            var mx = max(br, max(bg, bb))
            if mx > max_component_value:
                var s = max_component_value / mx
                br *= s
                bg *= s
                bb *= s
        beauty_out[i * 3 + 0] = br
        beauty_out[i * 3 + 1] = bg
        beauty_out[i * 3 + 2] = bb
        albedo_out[i * 3 + 0] = r.albedoR / w
        albedo_out[i * 3 + 1] = r.albedoG / w
        albedo_out[i * 3 + 2] = r.albedoB / w


# Pure functions over a contiguous byte buffer with a cursor index.
# Called by the Swift PbrtScanner after step 11a loaded the whole file
# into one buffer, enabling numeric parsing without buffer-boundary hazards.

@always_inline
fn _is_ws(b: UInt8) -> Bool:
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13)

@always_inline
fn _is_digit(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)

@export
def mojo_scan_int(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur >= len:
        return Int32(0)
    var negative = False
    if bytes[cur] == UInt8(45):       # '-'
        negative = True
        cur += 1
    if cur >= len or not _is_digit(bytes[cur]):
        return Int32(0)
    var value = Int32(0)
    while cur < len and _is_digit(bytes[cur]):
        value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
    if negative:
        value = -value
    cursor[0] = Int32(cur)
    result[0] = value
    return Int32(1)

@export
def mojo_scan_float(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur >= len:
        return Int32(0)

    # Remember whether a leading '-' was present for the sign fix-up below.
    # (scanInt treats '-0' as 0, so '-0.5' needs special handling.)
    var leading_negative = bytes[cur] == UInt8(45)

    # --- integer part (includes its own sign handling, mirrors scanInt) ---
    var int_negative = False
    if cur < len and bytes[cur] == UInt8(45):
        int_negative = True
        cur += 1
    var int_part = Int32(0)
    var int_seen = False
    while cur < len and _is_digit(bytes[cur]):
        int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
        int_seen = True
    if int_negative:
        int_part = -int_part

    var dval = Float64(int_part)

    # --- fractional part ---
    if cur < len and bytes[cur] == UInt8(46):   # '.'
        cur += 1
        var tenth = Float64(0.1)
        while cur < len and _is_digit(bytes[cur]):
            var d = Float64(Int32(bytes[cur]) - Int32(48))
            if dval < Float64(0.0):
                dval -= tenth * d
            else:
                dval += tenth * d
            tenth *= Float64(0.1)
            cur += 1
    elif not int_seen:
        return Int32(0)

    # --- exponent (mirrors calling scanInt for the exponent digits) ---
    if cur < len and bytes[cur] == UInt8(101):  # 'e'
        cur += 1
        while cur < len and _is_ws(bytes[cur]):  # scanInt skips ws
            cur += 1
        var exp_negative = False
        if cur < len and bytes[cur] == UInt8(45):
            exp_negative = True
            cur += 1
        var exp_val = Int32(0)
        while cur < len and _is_digit(bytes[cur]):
            exp_val = exp_val * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
        if exp_negative:
            exp_val = -exp_val
        var factor = Float64(1.0)
        var abs_exp = exp_val if exp_val >= 0 else -exp_val
        for _ in range(Int(abs_exp)):
            factor *= Float64(10.0)
        if exp_val < 0:
            dval /= factor
        else:
            dval *= factor

    var f = Float32(dval)
    if leading_negative and int_part == Int32(0):
        f = -f

    cursor[0] = Int32(cur)
    result[0] = f
    return Int32(1)

# Count floats in bytes[cursor..] without advancing the caller's cursor.
# Stops at ']', EOF, or any non-numeric token — same stopping condition as
# the per-float mojo_scan_float loop that parseReals() used to drive.
@export
def mojo_count_floats(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):   # optional '-'
            cur += 1
        var int_seen = False
        while cur < len and _is_digit(bytes[cur]):
            int_seen = True
            cur += 1
        if cur < len and bytes[cur] == UInt8(46):   # '.'
            cur += 1
            while cur < len and _is_digit(bytes[cur]):
                cur += 1
        elif not int_seen:
            break
        if cur < len and bytes[cur] == UInt8(101):  # 'e'
            cur += 1
            if cur < len and bytes[cur] == UInt8(45):
                cur += 1
            while cur < len and _is_digit(bytes[cur]):
                cur += 1
        count += Int32(1)
    return count

# Fill out[0..max_count) with floats parsed from bytes[*cursor..].
# Advances *cursor to the position where scanning stopped (at ']' or EOF).
# Returns the number of values written (== mojo_count_floats result when
# max_count >= that value).
@export
def mojo_scan_floats(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    max_count: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    var count = Int32(0)
    while count < max_count:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        var leading_negative = bytes[cur] == UInt8(45)
        var int_negative = False
        if cur < len and bytes[cur] == UInt8(45):
            int_negative = True
            cur += 1
        var int_part = Int32(0)
        var int_seen = False
        while cur < len and _is_digit(bytes[cur]):
            int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
            int_seen = True
        if int_negative:
            int_part = -int_part
        var dval = Float64(int_part)
        if cur < len and bytes[cur] == UInt8(46):
            cur += 1
            var tenth = Float64(0.1)
            while cur < len and _is_digit(bytes[cur]):
                var d = Float64(Int32(bytes[cur]) - Int32(48))
                if dval < Float64(0.0):
                    dval -= tenth * d
                else:
                    dval += tenth * d
                tenth *= Float64(0.1)
                cur += 1
        elif not int_seen:
            break
        if cur < len and bytes[cur] == UInt8(101):
            cur += 1
            while cur < len and _is_ws(bytes[cur]):
                cur += 1
            var exp_negative = False
            if cur < len and bytes[cur] == UInt8(45):
                exp_negative = True
                cur += 1
            var exp_val = Int32(0)
            while cur < len and _is_digit(bytes[cur]):
                exp_val = exp_val * Int32(10) + Int32(bytes[cur]) - Int32(48)
                cur += 1
            if exp_negative:
                exp_val = -exp_val
            var factor = Float64(1.0)
            var abs_exp = exp_val if exp_val >= 0 else -exp_val
            for _ in range(Int(abs_exp)):
                factor *= Float64(10.0)
            if exp_val < 0:
                dval /= factor
            else:
                dval *= factor
        var f = Float32(dval)
        if leading_negative and int_part == Int32(0):
            f = -f
        result[Int(count)] = f
        count += Int32(1)
    cursor[0] = Int32(cur)
    return count

# Count integers in bytes[cursor..] without advancing the caller's cursor.
@export
def mojo_count_ints(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):
            cur += 1
        if cur >= len or not _is_digit(bytes[cur]):
            break
        while cur < len and _is_digit(bytes[cur]):
            cur += 1
        count += Int32(1)
    return count

# Fill out[0..max_count) with integers from bytes[*cursor..].
# Advances *cursor to where scanning stopped.
@export
def mojo_scan_ints(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Int32, MutAnyOrigin],
    max_count: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    var count = Int32(0)
    while count < max_count:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        var negative = False
        if bytes[cur] == UInt8(45):
            negative = True
            cur += 1
        if cur >= len or not _is_digit(bytes[cur]):
            break
        var value = Int32(0)
        while cur < len and _is_digit(bytes[cur]):
            value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
        if negative:
            value = -value
        result[Int(count)] = value
        count += Int32(1)
    cursor[0] = Int32(cur)
    return count

# Skip leading whitespace, check if next byte == expected, consume it on match.
# Always advances *cursor past whitespace (even on failure).
# Returns 1 on success, 0 otherwise.
@export
def mojo_scan_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    cursor[0] = Int32(cur + 1)
    return Int32(1)

# Skip leading whitespace, check if next byte == expected (do NOT consume it).
# Always advances *cursor past whitespace.  Returns 1 on match, 0 otherwise.
@export
def mojo_peek_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    return Int32(1)

# Skip leading whitespace then read bytes into buf until a delimiter or EOF.
# Writes up to max_buf-1 bytes and null-terminates buf.
# Advances *cursor to the stopping position (at the delimiter, or at len).
# Returns bytes-written >= 0, or -1 if EOF was reached before any content.
@export
def mojo_scan_token(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    delims: UnsafePointer[UInt8, MutAnyOrigin],
    n_delims: Int32,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    max_buf: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len:
        if max_buf > 0:
            buf[0] = UInt8(0)
        return Int32(-1)
    var n = Int(n_delims)
    var written = Int32(0)
    while cur < len:
        var b = bytes[cur]
        var is_delim = False
        for i in range(n):
            if delims[i] == b:
                is_delim = True
                break
        if is_delim:
            break
        if written < max_buf - 1:
            buf[Int(written)] = b
        written += Int32(1)
        cur += 1
    var cap = Int(written) if Int(written) < Int(max_buf) - 1 else Int(max_buf) - 1
    if max_buf > 0:
        buf[cap] = UInt8(0)
    cursor[0] = Int32(cur)
    return written

# Read a complete quoted string: skip ws, consume '"', read content, consume '"'.
# Writes content (without quotes) to buf, null-terminates.
# Returns bytes written >= 0, or -1 if no opening '"' found.
@export
def mojo_parse_quoted_string(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    max_buf: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != UInt8(34):   # '"' = 34
        return Int32(-1)
    cur += 1  # opening '"'
    var written = Int32(0)
    while cur < len and bytes[cur] != UInt8(34):
        if written < max_buf - 1:
            buf[Int(written)] = bytes[cur]
        written += Int32(1)
        cur += 1
    if cur < len:
        cur += 1  # closing '"'
    if max_buf > 0:
        var cap = Int(written) if Int(written) < Int(max_buf) - 1 else Int(max_buf) - 1
        buf[cap] = UInt8(0)
    cursor[0] = Int32(cur)
    return written

# Read one pbrt parameter header: "type name" with optional leading '['.
# Skips leading whitespace. Writes null-terminated type and name to their buffers.
# Sets *is_array to 1 and consumes '[' if present, 0 otherwise.
# Returns 1 if a header was found, 0 if the next non-ws token is not '"'.
@export
def mojo_parse_param_header(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    type_buf: UnsafePointer[UInt8, MutAnyOrigin],
    type_max: Int32,
    name_buf: UnsafePointer[UInt8, MutAnyOrigin],
    name_max: Int32,
    is_array: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != UInt8(34):
        return Int32(0)
    cur += 1  # opening '"'
    # type: read until whitespace or '"'
    var t = Int32(0)
    while cur < len and not _is_ws(bytes[cur]) and bytes[cur] != UInt8(34):
        if t < type_max - 1:
            type_buf[Int(t)] = bytes[cur]
        t += Int32(1)
        cur += 1
    if type_max > 0:
        var cap = Int(t) if Int(t) < Int(type_max) - 1 else Int(type_max) - 1
        type_buf[cap] = UInt8(0)
    # skip separator whitespace
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    # name: read until '"'
    var n = Int32(0)
    while cur < len and bytes[cur] != UInt8(34):
        if n < name_max - 1:
            name_buf[Int(n)] = bytes[cur]
        n += Int32(1)
        cur += 1
    if name_max > 0:
        var cap = Int(n) if Int(n) < Int(name_max) - 1 else Int(name_max) - 1
        name_buf[cap] = UInt8(0)
    if cur < len and bytes[cur] == UInt8(34):
        cur += 1  # closing '"'
    # skip ws, check for '['
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur < len and bytes[cur] == UInt8(91):   # '[' = 91
        cur += 1
        is_array[0] = Int32(1)
    else:
        is_array[0] = Int32(0)
    cursor[0] = Int32(cur)
    return Int32(1)


# PbrtScanner in Mojo (step 19). Replaces PbrtScanner.swift.
# File is loaded into a heap buffer; all scanning delegates to the existing
# mojo_scan_* / mojo_parse_* primitives.

struct PbrtScanner_Mojo:
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var total_bytes: Int32
    var cursor: Int32
    var is_at_end: Int32


@always_inline
fn _scanner_cursor_ptr(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> UnsafePointer[Int32, MutAnyOrigin]:
    # cursor is the third field; layout: ptr(8) + total_bytes(4) + pad(4) + cursor(4)
    # Use alloc/copy-out/copy-back instead of field address arithmetic.
    # (Field-address trick is fragile; callers use _scanner_call instead.)
    return UnsafePointer[Int32, MutAnyOrigin]()


@always_inline
fn _scanner_call_int(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_int(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@always_inline
fn _scanner_call_float(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_float(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_new(path: UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]:
    var mode = alloc[UInt8](3)
    mode[0] = UInt8(114)   # 'r'
    mode[1] = UInt8(98)    # 'b'
    mode[2] = UInt8(0)
    var fp = external_call["fopen", UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]](path, mode)
    mode.free()
    var handle = alloc[PbrtScanner_Mojo](1)
    if not fp:
        handle[0].buffer = UnsafePointer[UInt8, MutAnyOrigin]()
        handle[0].total_bytes = Int32(0)
        handle[0].cursor = Int32(0)
        handle[0].is_at_end = Int32(1)
        return handle
    _ = external_call["fseek", Int32, UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp, Int64(0), Int32(2))
    var size = external_call["ftell", Int64, UnsafePointer[UInt8, MutAnyOrigin]](fp)
    _ = external_call["fseek", Int32, UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp, Int64(0), Int32(0))
    var buf = alloc[UInt8](Int(size) + 1)
    _ = external_call["fread", Int, UnsafePointer[UInt8, MutAnyOrigin], Int, Int, UnsafePointer[UInt8, MutAnyOrigin]](buf, 1, Int(size), fp)
    _ = external_call["fclose", Int32, UnsafePointer[UInt8, MutAnyOrigin]](fp)
    buf[Int(size)] = UInt8(0)
    handle[0].buffer = buf
    handle[0].total_bytes = Int32(size)
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


@export
fn mojo_scanner_new_from_bytes(bytes: UnsafePointer[UInt8, MutAnyOrigin], length: Int32) -> UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]:
    var handle = alloc[PbrtScanner_Mojo](1)
    var buf = alloc[UInt8](Int(length) + 1)
    for i in range(Int(length)):
        buf[i] = bytes[i]
    buf[Int(length)] = UInt8(0)
    handle[0].buffer = buf
    handle[0].total_bytes = length
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


@export
fn mojo_scanner_free(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]):
    if handle[0].buffer:
        handle[0].buffer.free()
    handle.free()


@export
fn mojo_scanner_is_at_end(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return handle[0].is_at_end


@export
fn mojo_scanner_scan_location(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return handle[0].cursor


@export
fn mojo_scanner_peek_char(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_peek_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_scan_char(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_scan_int(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    return _scanner_call_int(handle, result)


@export
fn mojo_scanner_scan_float(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    return _scanner_call_float(handle, result)


@export
fn mojo_scanner_count_floats(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return mojo_count_floats(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


@export
fn mojo_scanner_scan_floats(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], dst: UnsafePointer[Float32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_floats(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_count_ints(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return mojo_count_ints(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


@export
fn mojo_scanner_scan_ints(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], dst: UnsafePointer[Int32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_ints(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_parse_quoted_string(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_parse_quoted_string(handle[0].buffer, handle[0].total_bytes, cur, buf, max_buf)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_parse_param_header(
    handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
    type_buf: UnsafePointer[UInt8, MutAnyOrigin], type_max: Int32,
    name_buf: UnsafePointer[UInt8, MutAnyOrigin], name_max: Int32,
    is_array: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_parse_param_header(handle[0].buffer, handle[0].total_bytes, cur,
                                      type_buf, type_max, name_buf, name_max, is_array)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@export
fn mojo_scanner_scan_token(
    handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
    delims: UnsafePointer[UInt8, MutAnyOrigin], n_delims: Int32,
    buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32,
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_token(handle[0].buffer, handle[0].total_bytes, cur,
                              delims, n_delims, buf, max_buf)
    handle[0].cursor = cur[0]
    if ret < 0:
        handle[0].is_at_end = Int32(1)
    cur.free()
    return ret


# Matrix math (step 13a). 4x4 matrices are 16 Float32 in column-major order:
# flat[col*4 + row] = matrix[row, col], matching Transform.columnMajorFloats().

fn _write_identity(result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    for i in range(16):
        result[i] = Float32(0)
    result[0] = Float32(1)
    result[5] = Float32(1)
    result[10] = Float32(1)
    result[15] = Float32(1)
    return Int32(0)


@export
fn mojo_matrix_multiply(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
):
    # result[i, j] = sum_k a[i, k] * b[k, j]   (column-major)
    for j in range(4):
        for i in range(4):
            var s = Float32(0)
            for k in range(4):
                s += a[k * 4 + i] * b[j * 4 + k]
            result[j * 4 + i] = s


@export
fn mojo_matrix_invert(
    m: UnsafePointer[Float32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
) -> Int32:
    # Gauss-Jordan elimination with full pivoting (mirrors Matrix.invert).
    # On a singular matrix, writes the identity and returns 0.
    var minv = InlineArray[Float32, 16](fill=0)
    for i in range(16):
        minv[i] = m[i]
    var indxc = InlineArray[Int, 4](fill=0)
    var indxr = InlineArray[Int, 4](fill=0)
    var ipiv = InlineArray[Int, 4](fill=0)

    for iteration in range(4):
        var big = Float32(0)
        var irow = 0
        var icol = 0
        for row in range(4):
            if ipiv[row] != 1:
                for col in range(4):
                    if ipiv[col] == 0:
                        var v = minv[col * 4 + row]
                        if v < Float32(0):
                            v = -v
                        if v >= big:
                            big = v
                            irow = row
                            icol = col
                    elif ipiv[col] > 1:
                        return _write_identity(result)
        ipiv[icol] += 1
        if irow != icol:
            for colIndex in range(4):
                var tmp = minv[colIndex * 4 + irow]
                minv[colIndex * 4 + irow] = minv[colIndex * 4 + icol]
                minv[colIndex * 4 + icol] = tmp
        indxr[iteration] = irow
        indxc[iteration] = icol
        if minv[icol * 4 + icol] == Float32(0):
            return _write_identity(result)
        var pivinv = Float32(1) / minv[icol * 4 + icol]
        minv[icol * 4 + icol] = Float32(1)
        for colIndex in range(4):
            minv[colIndex * 4 + icol] *= pivinv
        for rowIndex in range(4):
            if rowIndex != icol:
                var save = minv[icol * 4 + rowIndex]
                minv[icol * 4 + rowIndex] = Float32(0)
                for colIndex in range(4):
                    minv[colIndex * 4 + rowIndex] -= minv[colIndex * 4 + icol] * save

    for ci in range(4):
        var colIndex = 3 - ci
        if indxr[colIndex] != indxc[colIndex]:
            var rs = indxr[colIndex]
            var cs = indxc[colIndex]
            for rowIndex in range(4):
                var tmp = minv[rs * 4 + rowIndex]
                minv[rs * 4 + rowIndex] = minv[cs * 4 + rowIndex]
                minv[cs * 4 + rowIndex] = tmp

    for i in range(16):
        result[i] = minv[i]
    return Int32(1)


# Bulk geometry transform (step 13b).
# Points are 4 floats each (SIMD4 layout: x,y,z,w=1). Normals are 3 floats each.
# matrix / inv_matrix are 16-float column-major (flat[col*4+row] = m[row,col]).

@export
fn mojo_transform_points(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    points_in: UnsafePointer[Float32, MutAnyOrigin],
    count: Int32,
    points_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var m0 = matrix[0];  var m1 = matrix[1];  var m2 = matrix[2];  var m3 = matrix[3]
    var m4 = matrix[4];  var m5 = matrix[5];  var m6 = matrix[6];  var m7 = matrix[7]
    var m8 = matrix[8];  var m9 = matrix[9];  var m10 = matrix[10]; var m11 = matrix[11]
    var m12 = matrix[12]; var m13 = matrix[13]; var m14 = matrix[14]; var m15 = matrix[15]
    for i in range(Int(count)):
        var b = i * 4
        var px = points_in[b];  var py = points_in[b+1];  var pz = points_in[b+2]
        var rx = m0*px + m4*py + m8*pz + m12
        var ry = m1*px + m5*py + m9*pz + m13
        var rz = m2*px + m6*py + m10*pz + m14
        var rw = m3*px + m7*py + m11*pz + m15
        if rw != Float32(1) and rw != Float32(0):
            var inv_rw = Float32(1) / rw
            rx *= inv_rw; ry *= inv_rw; rz *= inv_rw
        points_out[b] = rx;  points_out[b+1] = ry;  points_out[b+2] = rz;  points_out[b+3] = Float32(1)


@export
fn mojo_transform_normals(
    inv_matrix: UnsafePointer[Float32, MutAnyOrigin],
    normals_in: UnsafePointer[Float32, MutAnyOrigin],
    count: Int32,
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
):
    # Normals transform by the transpose of the inverse 3×3.
    # result[i] = sum_j inv[j*4+i] * n[j]  for i,j in 0..2
    var i0 = inv_matrix[0]; var i1 = inv_matrix[1]; var i2 = inv_matrix[2]
    var i4 = inv_matrix[4]; var i5 = inv_matrix[5]; var i6 = inv_matrix[6]
    var i8 = inv_matrix[8]; var i9 = inv_matrix[9]; var i10 = inv_matrix[10]
    for i in range(Int(count)):
        var b = i * 3
        var nx = normals_in[b];  var ny = normals_in[b+1];  var nz = normals_in[b+2]
        normals_out[b]   = i0*nx + i4*ny + i8*nz
        normals_out[b+1] = i1*nx + i5*ny + i9*nz
        normals_out[b+2] = i2*nx + i6*ny + i10*nz


# Joint bilateral denoiser guided by albedo (step 17).
# beauty and albedo are width*height*3 float arrays (R,G,B interleaved, row-major).
# output must hold width*height*3 floats. radius is the filter half-width (window = 2r+1).
# sigma_s: spatial Gaussian sigma; sigma_r: albedo-difference Gaussian sigma.
# Spatial weights are precomputed; spatial+range are merged into one exp() per neighbor.
@export
fn mojo_denoise(
    beauty: UnsafePointer[Float32, MutAnyOrigin],
    albedo: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    radius: Int32,
    sigma_s: Float32,
    sigma_r: Float32,
):
    var w = Int(width)
    var h = Int(height)
    var r = Int(radius)
    var inv2ss = Float32(1) / (Float32(2) * sigma_s * sigma_s)
    var inv2sr = Float32(1) / (Float32(2) * sigma_r * sigma_r)
    var diam = 2 * r + 1

    # Precompute spatial weights for the (2r+1)×(2r+1) window.
    var sw = alloc[Float32](diam * diam)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            var dist2 = Float32(dx * dx + dy * dy)
            sw[(dy + r) * diam + (dx + r)] = exp(-dist2 * inv2ss)

    for py in range(h):
        for px in range(w):
            var ci = (py * w + px) * 3
            var a0r = albedo[ci + 0]
            var a0g = albedo[ci + 1]
            var a0b = albedo[ci + 2]

            var acc_r = Float32(0)
            var acc_g = Float32(0)
            var acc_b = Float32(0)
            var acc_w = Float32(0)

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    var nx = px + dx
                    var ny = py + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    var ni = (ny * w + nx) * 3
                    var dar = albedo[ni + 0] - a0r
                    var dag = albedo[ni + 1] - a0g
                    var dab = albedo[ni + 2] - a0b
                    var albedo_dist2 = dar * dar + dag * dag + dab * dab
                    # Merge spatial and range into one exp call.
                    var wt = sw[(dy + r) * diam + (dx + r)] * exp(-albedo_dist2 * inv2sr)
                    acc_r += beauty[ni + 0] * wt
                    acc_g += beauty[ni + 1] * wt
                    acc_b += beauty[ni + 2] * wt
                    acc_w += wt

            if acc_w > Float32(0):
                output[ci + 0] = acc_r / acc_w
                output[ci + 1] = acc_g / acc_w
                output[ci + 2] = acc_b / acc_w
            else:
                output[ci + 0] = beauty[ci + 0]
                output[ci + 1] = beauty[ci + 1]
                output[ci + 2] = beauty[ci + 2]

    sw.free()


# Write a float RGB buffer to an EXR file via the OpenImageIO bridge (step 18).
# pixels must hold width*height*3 floats (R,G,B interleaved, row-major).
# filename must be a null-terminated UTF-8 string.
# Returns 1 on success, 0 on failure.
@export
fn mojo_write_exr(
    pixels: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    filename: UnsafePointer[UInt8, MutAnyOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    return external_call["write_exr_rgb", Int32,
        UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[Float32, MutAnyOrigin],
        Int32, Int32, Int32, Int32,
    ](filename, pixels, width, height, tile_w, tile_h)


@export
fn mojo_gpu_free_scene(handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin]):
    if not handlePtr:
        return
    handlePtr.destroy_pointee()
    handlePtr.bitcast[GpuSceneHandle]().free()
    print("GPU: Scene resources freed")
