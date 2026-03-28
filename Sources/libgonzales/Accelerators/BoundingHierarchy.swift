import mojoKernel

final class BoundingHierarchy: Boundable, Intersectable, @unchecked Sendable {

        let bvh2NodesPointer: UnsafeMutablePointer<BVH2Node>
        let bvh2NodesCount: Int
        let primIdsPointer: UnsafeMutablePointer<PrimId>
        let primIdsCount: Int

        init(primitives: [IntersectablePrimitive], bvh2Nodes: [BVH2Node]) {
                self.bvh2NodesCount = bvh2Nodes.count
                self.bvh2NodesPointer = UnsafeMutablePointer<BVH2Node>.allocate(capacity: max(bvh2Nodes.count, 1))
                if !bvh2Nodes.isEmpty {
                        self.bvh2NodesPointer.initialize(from: bvh2Nodes, count: bvh2Nodes.count)
                }

                var ids = [PrimId]()
                for primitive in primitives {
                        switch primitive {
                        case .geometricPrimitive(let geometricPrimitive):
                                if case .triangle(let triangle) = geometricPrimitive.shape {
                                        // Pack meshIndex high, triIndex low into id2 so Mojo can intersect
                                        let packed = (triangle.meshIndex << 32) | (triangle.triangleIndex / 3)
                                        let primId = PrimId(
                                                id1: geometricPrimitive.idx, id2: packed, type: .geometricPrimitive)
                                        ids.append(primId)
                                } else {
                                        let primId = PrimId(
                                                id1: geometricPrimitive.idx, id2: -1, type: .geometricPrimitive)
                                        ids.append(primId)
                                }
                        case .triangle(let triangle):
                                let primId = PrimId(
                                        id1: triangle.meshIndex, id2: triangle.triangleIndex, type: .triangle)
                                ids.append(primId)
                        case .transformedPrimitive(let transformedPrimitive):
                                let primId = PrimId(
                                        id1: transformedPrimitive.idx, id2: -1, type: .transformedPrimitive)
                                ids.append(primId)
                        case .areaLight(let areaLight):
                                if case .triangle(let triangle) = areaLight.shape {
                                        let packed = (triangle.meshIndex << 32) | (triangle.triangleIndex / 3)
                                        let primId = PrimId(id1: areaLight.idx, id2: packed, type: .areaLight)
                                        ids.append(primId)
                                } else {
                                        let primId = PrimId(id1: areaLight.idx, id2: -1, type: .areaLight)
                                        ids.append(primId)
                                }
                        }
                }
                self.primIdsCount = ids.count
                self.primIdsPointer = UnsafeMutablePointer<PrimId>.allocate(capacity: max(ids.count, 1))
                if !ids.isEmpty {
                        self.primIdsPointer.initialize(from: ids, count: ids.count)
                }

                print("LAYOUT: BVH2Node     stride=\(MemoryLayout<BVH2Node>.stride) " +
                      "size=\(MemoryLayout<BVH2Node>.size) " +
                      "align=\(MemoryLayout<BVH2Node>.alignment)")
                print("LAYOUT: BVH2 nodes=\(bvh2Nodes.count) prims=\(ids.count)")
                print("LAYOUT: BVH2 memory=\(bvh2Nodes.count * MemoryLayout<BVH2Node>.stride) bytes")
        }


        deinit {
                if bvh2NodesCount > 0 {
                        bvh2NodesPointer.deinitialize(count: bvh2NodesCount)
                }
                bvh2NodesPointer.deallocate()
                if primIdsCount > 0 {
                        primIdsPointer.deinitialize(count: primIdsCount)
                }
                primIdsPointer.deallocate()
        }

        // --- Occlusion Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Bool {
                if bvh2NodesCount == 0 { return false }

                return scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        var desc = SceneDescriptor2_C(
                                bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer).assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                primIds: UnsafeRawPointer(primIdsPointer).assumingMemoryBound(to: PrimId_C.self),
                                meshes: meshesPtr.baseAddress,
                                meshCount: Int64(scene.meshesC.count)
                        )
                        var rayC = Ray_C(
                                orgX: Float(ray.origin.x), orgY: Float(ray.origin.y), orgZ: Float(ray.origin.z),
                                dirX: Float(ray.direction.x), dirY: Float(ray.direction.y), dirZ: Float(ray.direction.z)
                        )
                        var result = Intersection_C()
                        withUnsafePointer(to: &desc) { descP in
                                withUnsafePointer(to: &rayC) { rayP in
                                        withUnsafeMutablePointer(to: &result) { resP in
                                                mojo_traverse_bvh2(descP, rayP, Float(tHit), resP)
                                        }
                                }
                        }
                        if result.hit != 0 {
                                tHit = Real(result.tHit)
                                return true
                        }
                        return false
                }
        }

        // --- Closest Hit Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if bvh2NodesCount == 0 { return nil }

                let result = scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        var desc = SceneDescriptor2_C(
                                bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer).assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                primIds: UnsafeRawPointer(primIdsPointer).assumingMemoryBound(to: PrimId_C.self),
                                meshes: meshesPtr.baseAddress,
                                meshCount: Int64(scene.meshesC.count)
                        )
                        var rayC = Ray_C(
                                orgX: Float(ray.origin.x), orgY: Float(ray.origin.y), orgZ: Float(ray.origin.z),
                                dirX: Float(ray.direction.x), dirY: Float(ray.direction.y), dirZ: Float(ray.direction.z)
                        )
                        var result = Intersection_C()
                        withUnsafePointer(to: &desc) { descP in
                                withUnsafePointer(to: &rayC) { rayP in
                                        withUnsafeMutablePointer(to: &result) { resP in
                                                mojo_traverse_bvh2(descP, rayP, Float(tHit), resP)
                                        }
                                }
                        }
                        return result
                }

                if result.hit != 0 {
                        let id1 = Int(result.primId.id1)
                        let rawId2 = Int(result.primId.id2)

                        let type: PrimType
                        if result.primId.type == 0 {
                                type = .triangle
                        } else if result.primId.type == 1 {
                                type = .geometricPrimitive
                        } else {
                                type = .areaLight
                        }

                        let meshIdx: Int
                        let triIdx: Int
                        if type == .geometricPrimitive || type == .areaLight {
                                meshIdx = rawId2 >> 32
                                triIdx = rawId2 & 0xFFFFFFFF
                        } else {
                                meshIdx = id1
                                triIdx = rawId2
                        }

                        let data = TriangleIntersection(
                                primId: PrimId(id1: meshIdx, id2: triIdx, type: .triangle),
                                tValue: Real(result.tHit),
                                barycentric0: Real(1.0 - result.u - result.v),
                                barycentric1: Real(result.u),
                                barycentric2: Real(result.v)
                        )
                        tHit = Real(result.tHit)
                        return scene.computeSurfaceInteraction(primId: PrimId(id1: id1, id2: rawId2, type: type), data: data, worldRay: ray)
                }

                return nil
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return worldBound(scene: scene)
        }

        func worldBound(scene _: Scene) -> Bounds3f {
                if bvh2NodesCount == 0 {
                        return Bounds3f()
                } else {
                        let root = bvh2NodesPointer[0]
                        return Bounds3f(
                                first: Point3(x: Real(root.boundsMinX), y: Real(root.boundsMinY), z: Real(root.boundsMinZ)),
                                second: Point3(x: Real(root.boundsMaxX), y: Real(root.boundsMaxY), z: Real(root.boundsMaxZ))
                        )
                }
        }
}

enum PrimType: UInt8 {
        case triangle
        case geometricPrimitive
        case transformedPrimitive
        case areaLight
}

struct PrimId {
        init() {
                id1 = 0
                id2 = 0
                type = .triangle
        }
        init(id1: Int, id2: Int, type: PrimType) {
                self.id1 = id1
                self.id2 = id2
                self.type = type
        }

        let id1: Int
        let id2: Int
        let type: PrimType
}
