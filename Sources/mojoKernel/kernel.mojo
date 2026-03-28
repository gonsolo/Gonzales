from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from std.memory import alloc

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
                var mesh_idx: Int = -1
                var base_vidx: Int = -1

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
        var dummyId = PrimId_C(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0)
        resultPtr[0] = Intersection_C(dummyId, tMax, 0.0, 0.0, Int8(0), 0, 0, 0)


# ── CPU entry point (called from Swift via C FFI) ─────────────────────────────

@export
fn mojo_traverse_bvh2(scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin], rayPtr: UnsafePointer[Ray_C, MutAnyOrigin], tMax: Float32, resultPtr: UnsafePointer[Intersection_C, MutAnyOrigin]):
    var scene = scenePtr[0]
    var ray = rayPtr[0]
    traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, ray, tMax, resultPtr)


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
) -> UnsafePointer[GpuSceneHandle, MutAnyOrigin]:
    try:
        var ctx = DeviceContext()

        # Check GPU memory
        var mem_info = ctx.get_memory_info()
        var free_bytes = mem_info[0]
        var total_bytes = mem_info[1]

        var bvh_bytes = Int(bvh2NodesCount) * 32  # sizeof(BVH2Node) = 32
        var prim_bytes = Int(primIdsCount) * 24    # sizeof(PrimId_C) = 24 (8+8+1+7 padding)
        var mesh_struct_bytes = Int(meshCount) * 24  # sizeof(TriangleMesh_C) = 3 pointers

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

        ctx.synchronize()

        # Allocate handle on heap
        var handle = alloc[GpuSceneHandle](1)
        handle.init_pointee_move(GpuSceneHandle(
            ctx=ctx^,
            bvh2Nodes_buf=bvh_buf^,
            primIds_buf=prim_buf^,
            meshes_buf=meshes_buf^,
            mesh_count=Int(meshCount),
            points_bufs=points_bufs^,
            faceIndices_bufs=face_bufs^,
            vertexIndices_bufs=vert_bufs^,
        ))

        print("GPU: Scene uploaded successfully")
        return handle.bitcast[GpuSceneHandle]()
    except e:
        print("GPU: Failed to upload scene: " + String(e))
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
        var result_bytes = n * 40  # sizeof(Intersection_C) = PrimId(24) + f32*3(12) + i8*4(4) = 40
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

        # Copy results back to host
        with result_buf.map_to_host() as host_buf:
            var src = host_buf.unsafe_ptr()
            var dst = results.bitcast[UInt8]()
            for i in range(result_bytes):
                dst[i] = src[i]
    except e:
        print("GPU: Batch traversal failed: " + String(e))


@export
fn mojo_gpu_free_scene(handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin]):
    if not handlePtr:
        return
    handlePtr.destroy_pointee()
    handlePtr.bitcast[GpuSceneHandle]().free()
    print("GPU: Scene resources freed")
