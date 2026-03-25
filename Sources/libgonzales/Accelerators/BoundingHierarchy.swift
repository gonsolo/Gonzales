import mojoKernel

struct Stack128 {
    var e00 = 0, e01 = 0, e02 = 0, e03 = 0, e04 = 0, e05 = 0, e06 = 0, e07 = 0
    var e08 = 0, e09 = 0, e10 = 0, e11 = 0, e12 = 0, e13 = 0, e14 = 0, e15 = 0
    var e16 = 0, e17 = 0, e18 = 0, e19 = 0, e20 = 0, e21 = 0, e22 = 0, e23 = 0
    var e24 = 0, e25 = 0, e26 = 0, e27 = 0, e28 = 0, e29 = 0, e30 = 0, e31 = 0
    var e32 = 0, e33 = 0, e34 = 0, e35 = 0, e36 = 0, e37 = 0, e38 = 0, e39 = 0
    var e40 = 0, e41 = 0, e42 = 0, e43 = 0, e44 = 0, e45 = 0, e46 = 0, e47 = 0
    var e48 = 0, e49 = 0, e50 = 0, e51 = 0, e52 = 0, e53 = 0, e54 = 0, e55 = 0
    var e56 = 0, e57 = 0, e58 = 0, e59 = 0, e60 = 0, e61 = 0, e62 = 0, e63 = 0
    var e64 = 0, e65 = 0, e66 = 0, e67 = 0, e68 = 0, e69 = 0, e70 = 0, e71 = 0
    var e72 = 0, e73 = 0, e74 = 0, e75 = 0, e76 = 0, e77 = 0, e78 = 0, e79 = 0
    var e80 = 0, e81 = 0, e82 = 0, e83 = 0, e84 = 0, e85 = 0, e86 = 0, e87 = 0
    var e88 = 0, e89 = 0, e90 = 0, e91 = 0, e92 = 0, e93 = 0, e94 = 0, e95 = 0
    var e96 = 0, e97 = 0, e98 = 0, e99 = 0, e100 = 0, e101 = 0, e102 = 0, e103 = 0
    var e104 = 0, e105 = 0, e106 = 0, e107 = 0, e108 = 0, e109 = 0, e110 = 0, e111 = 0
    var e112 = 0, e113 = 0, e114 = 0, e115 = 0, e116 = 0, e117 = 0, e118 = 0, e119 = 0
    var e120 = 0, e121 = 0, e122 = 0, e123 = 0, e124 = 0, e125 = 0, e126 = 0, e127 = 0

    subscript(index: Int) -> Int {
        @inline(__always) get {
            return withUnsafePointer(to: self) { ptr in
                UnsafeRawPointer(ptr).assumingMemoryBound(to: Int.self)[index]
            }
        }
        @inline(__always) set {
            withUnsafeMutablePointer(to: &self) { ptr in
                UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Int.self)[index] = newValue
            }
        }
    }
}



final class BoundingHierarchy: Boundable, Intersectable, @unchecked Sendable {

        let nodesPointer: UnsafeMutablePointer<BoundingHierarchyNode>
        let nodesCount: Int
        let primIdsPointer: UnsafeMutablePointer<PrimId>
        let primIdsCount: Int

        init(primitives: [IntersectablePrimitive], nodes: [BoundingHierarchyNode]) {
                self.nodesCount = nodes.count
                self.nodesPointer = UnsafeMutablePointer<BoundingHierarchyNode>.allocate(capacity: nodes.count)
                self.nodesPointer.initialize(from: nodes, count: nodes.count)
                
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
                                let primId = PrimId(id1: areaLight.idx, id2: -1, type: .areaLight)
                                ids.append(primId)
                        }
                }
                self.primIdsCount = ids.count
                self.primIdsPointer = UnsafeMutablePointer<PrimId>.allocate(capacity: ids.count)
                self.primIdsPointer.initialize(from: ids, count: ids.count)
                
                // Count PrimId types
                var typeCount = [Int](repeating: 0, count: 4)
                for i in 0..<ids.count {
                        typeCount[Int(ids[i].type.rawValue)] += 1
                }
                print("LAYOUT: PrimId types: triangle=\(typeCount[0]) geoPrim=\(typeCount[1]) xform=\(typeCount[2]) areaLight=\(typeCount[3]) total=\(ids.count)")

                // One-time struct layout validation (Mojo expected sizes computed from struct definitions)
                print("LAYOUT: BVHNode      stride=\(MemoryLayout<BoundingHierarchyNode>.stride) size=\(MemoryLayout<BoundingHierarchyNode>.size) align=\(MemoryLayout<BoundingHierarchyNode>.alignment)")
                print("LAYOUT: PrimId       stride=\(MemoryLayout<PrimId>.stride) size=\(MemoryLayout<PrimId>.size)")
                print("LAYOUT: PrimId_C     stride=\(MemoryLayout<PrimId_C>.stride) size=\(MemoryLayout<PrimId_C>.size)")
                print("LAYOUT: SceneDesc_C  stride=\(MemoryLayout<SceneDescriptor_C>.stride) size=\(MemoryLayout<SceneDescriptor_C>.size)")
                print("LAYOUT: Ray_C        stride=\(MemoryLayout<Ray_C>.stride) size=\(MemoryLayout<Ray_C>.size)")
                print("LAYOUT: Intersection stride=\(MemoryLayout<Intersection_C>.stride) size=\(MemoryLayout<Intersection_C>.size)")
                print("LAYOUT: TriMesh_C    stride=\(MemoryLayout<TriangleMesh_C>.stride) size=\(MemoryLayout<TriangleMesh_C>.size)")
                if nodesCount > 0 {
                        let n = nodesPointer[0]
                        print("LAYOUT: root pMinX=\(n.pMinX)")
                        print("LAYOUT: root pMaxX=\(n.pMaxX)")
                        print("LAYOUT: root pMinY=\(n.pMinY)")
                        print("LAYOUT: root pMaxY=\(n.pMaxY)")
                        print("LAYOUT: root pMinZ=\(n.pMinZ)")
                        print("LAYOUT: root pMaxZ=\(n.pMaxZ)")
                        print("LAYOUT: root childNodes=\(n.childNodes)")
                        print("LAYOUT: root primCounts=\(n.primitiveCounts)")
                        // Test AABB intersection with a realistic diagonal ray
                        let testRay = Ray_C(orgX: 0.1, orgY: 0.5, orgZ: 0.5, dirX: 0.1, dirY: -0.2, dirZ: -0.8)
                        
                        var testRayMut2 = testRay
                        let mojoMask = withUnsafePointer(to: &testRayMut2) { rayP in
                                mojo_test_intersect(UnsafeRawPointer(nodesPointer), rayP, 1e30)
                        }
                        // Run Swift AABB test on same data
                        let rdirX = SIMD8<Float>(repeating: 1.0 / testRay.dirX)
                        let rdirY = SIMD8<Float>(repeating: 1.0 / testRay.dirY)
                        let rdirZ = SIMD8<Float>(repeating: 1.0 / testRay.dirZ)
                        let orgRdirX = SIMD8<Float>(repeating: testRay.orgX * (1.0 / testRay.dirX))
                        let orgRdirY = SIMD8<Float>(repeating: testRay.orgY * (1.0 / testRay.dirY))
                        let orgRdirZ = SIMD8<Float>(repeating: testRay.orgZ * (1.0 / testRay.dirZ))
                        let precomp = BoundingHierarchyNode.RayAABBPrecomputed(
                                rdirX: rdirX, rdirY: rdirY, rdirZ: rdirZ,
                                orgRdirX: orgRdirX, orgRdirY: orgRdirY, orgRdirZ: orgRdirZ,
                                nearXIsMin: (1.0 / testRay.dirX) >= 0,
                                nearYIsMin: (1.0 / testRay.dirY) >= 0,
                                nearZIsMin: (1.0 / testRay.dirZ) >= 0
                        )
                        var swiftMask: UInt8 = 0
                        let (_, swiftResult) = n.intersect8(ray: precomp, tHit: 1e30)
                        for i in 0..<8 { if swiftResult[i] { swiftMask |= UInt8(1 << i) } }
                        print("LAYOUT: AABB test ray=(0.1,0.5,0.5)->(0.1,-0.2,-0.8) SwiftMask=\(String(swiftMask, radix: 2)) MojoMask=\(String(mojoMask, radix: 2))")
                        if swiftMask != UInt8(mojoMask) {
                                print("LAYOUT: AABB MASK MISMATCH!")
                        }
                }
        }

        deinit {
                nodesPointer.deinitialize(count: nodesCount)
                nodesPointer.deallocate()
                primIdsPointer.deinitialize(count: primIdsCount)
                primIdsPointer.deallocate()
        }

        // Evaluate leaf primitives, returns true if any hit found
        @inline(__always)
        private func evaluateLeaf(
                node: BoundingHierarchyNode, childIndex: Int,
                scene: Scene, ray: Ray, tHit: inout Float
        ) -> Bool {
                let count = Int(node.primitiveCounts[childIndex])
                let offset = Int(node.primitiveOffsets[childIndex])
                var found = false
                var j = 0
                while j < count {
                        if scene.intersect(
                                primId: primIdsPointer[offset + j],
                                ray: ray, tHit: &tHit) {
                                found = true
                        }
                        j += 1
                }
                return found
        }

        // --- Occlusion Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Bool {
                if nodesCount == 0 { return false }
                
                return scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        var desc = SceneDescriptor_C(
                                bvhNodes: UnsafeRawPointer(nodesPointer),
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
                                                mojo_traverse(descP, rayP, Float(tHit), resP)
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

        // --- Closest Hit Query (Embree-style hit-count specialized) ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if nodesCount == 0 { return nil }

                let result = scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        var desc = SceneDescriptor_C(
                                bvhNodes: UnsafeRawPointer(nodesPointer),
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
                                                mojo_traverse(descP, rayP, Float(tHit), resP)
                                        }
                                }
                        }
                        return result
                }

                if result.hit != 0 {
                        let id1 = Int(result.primId.id1)
                        let id2 = Int(result.primId.id2)
                        let type = result.primId.type == 0 ? PrimType.triangle : PrimType.geometricPrimitive
                        
                        let data = TriangleIntersection(
                                primId: PrimId(id1: id1, id2: id2, type: type),
                                tValue: Real(result.tHit),
                                barycentric0: Real(1.0 - result.u - result.v),
                                barycentric1: Real(result.u),
                                barycentric2: Real(result.v)
                        )
                        tHit = Real(result.tHit)
                        return scene.computeSurfaceInteraction(primId: PrimId(id1: id1, id2: id2, type: type), data: data, worldRay: ray)
                }
                
                return nil
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return worldBound(scene: scene)
        }

        func worldBound(scene _: Scene) -> Bounds3f {
                if nodesCount == 0 {
                        return Bounds3f()
                } else {
                        let n = nodesPointer[0]
                        var totalBound = Bounds3f()
                        for i in 0..<8 {
                                if n.pMinX[i] != Float.infinity {
                                        let childBound = Bounds3f(
                                                first: Point3(x: Real(n.pMinX[i]), y: Real(n.pMinY[i]), z: Real(n.pMinZ[i])),
                                                second: Point3(x: Real(n.pMaxX[i]), y: Real(n.pMaxY[i]), z: Real(n.pMaxZ[i]))
                                        )
                                        totalBound = union(first: totalBound, second: childBound)
                                }
                        }
                        return totalBound
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
