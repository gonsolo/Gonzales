from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt, cos, sin
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
    var _pad0: Int32
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
fn mojo_gpu_free_scene(handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin]):
    if not handlePtr:
        return
    handlePtr.destroy_pointee()
    handlePtr.bitcast[GpuSceneHandle]().free()
    print("GPU: Scene resources freed")
