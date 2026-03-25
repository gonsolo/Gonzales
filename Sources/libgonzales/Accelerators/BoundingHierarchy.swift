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
                                let primId = PrimId(
                                        id1: geometricPrimitive.idx, id2: -1, type: .geometricPrimitive)
                                ids.append(primId)
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

                // Precompute ray constants (Embree-style)
                let rdirX = SIMD8<Float>(repeating: Float(ray.inverseDirection.x))
                let rdirY = SIMD8<Float>(repeating: Float(ray.inverseDirection.y))
                let rdirZ = SIMD8<Float>(repeating: Float(ray.inverseDirection.z))
                let orgRdirX = SIMD8<Float>(repeating: Float(ray.origin.x) * Float(ray.inverseDirection.x))
                let orgRdirY = SIMD8<Float>(repeating: Float(ray.origin.y) * Float(ray.inverseDirection.y))
                let orgRdirZ = SIMD8<Float>(repeating: Float(ray.origin.z) * Float(ray.inverseDirection.z))
                let nearXIsMin = ray.inverseDirection.x >= 0
                let nearYIsMin = ray.inverseDirection.y >= 0
                let nearZIsMin = ray.inverseDirection.z >= 0

                var localTHit = Float(tHit)
                var stack = Stack128()
                var toVisit = 0
                var current = 0

                while true {
                        let node = nodesPointer[current]
                        let (_, hitMask) = node.intersect8(
                                rdirX: rdirX, rdirY: rdirY, rdirZ: rdirZ,
                                orgRdirX: orgRdirX, orgRdirY: orgRdirY, orgRdirZ: orgRdirZ,
                                nearXIsMin: nearXIsMin, nearYIsMin: nearYIsMin, nearZIsMin: nearZIsMin,
                                tHit: localTHit)

                        var mask: UInt8 = 0
                        if hitMask[0] { mask |= 1 }
                        if hitMask[1] { mask |= 2 }
                        if hitMask[2] { mask |= 4 }
                        if hitMask[3] { mask |= 8 }
                        if hitMask[4] { mask |= 16 }
                        if hitMask[5] { mask |= 32 }
                        if hitMask[6] { mask |= 64 }
                        if hitMask[7] { mask |= 128 }
                        while mask != 0 {
                                let i = Int(mask.trailingZeroBitCount)
                                mask &= mask &- 1 // clear lowest set bit (bscf)

                                if node.primitiveCounts[i] > 0 { // leaf
                                        var realTHit = Real(localTHit)
                                        let count = Int(node.primitiveCounts[i])
                                        let offset = Int(node.primitiveOffsets[i])
                                        var j = 0
                                        while j < count {
                                                if scene.intersect(
                                                        primId: primIdsPointer[offset + j],
                                                        ray: ray,
                                                        tHit: &realTHit) {
                                                        tHit = realTHit
                                                        return true
                                                }
                                                j += 1
                                        }
                                        localTHit = Float(realTHit)
                                } else { // interior
                                        let childIdx = Int(node.childNodes[i])
                                        if childIdx >= 0 {
                                                stack[toVisit] = childIdx
                                                toVisit += 1
                                        }
                                }
                        }

                        if toVisit == 0 { break }
                        toVisit -= 1
                        current = stack[toVisit]
                }
                
                tHit = Real(localTHit)
                return false
        }

        // --- Closest Hit Query (Embree-style hit-count specialized) ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if nodesCount == 0 { return nil }

                // Precompute ray constants (Embree-style)
                let rdirX = SIMD8<Float>(repeating: Float(ray.inverseDirection.x))
                let rdirY = SIMD8<Float>(repeating: Float(ray.inverseDirection.y))
                let rdirZ = SIMD8<Float>(repeating: Float(ray.inverseDirection.z))
                let orgRdirX = SIMD8<Float>(repeating: Float(ray.origin.x) * Float(ray.inverseDirection.x))
                let orgRdirY = SIMD8<Float>(repeating: Float(ray.origin.y) * Float(ray.inverseDirection.y))
                let orgRdirZ = SIMD8<Float>(repeating: Float(ray.origin.z) * Float(ray.inverseDirection.z))
                let nearXIsMin = ray.inverseDirection.x >= 0
                let nearYIsMin = ray.inverseDirection.y >= 0
                let nearZIsMin = ray.inverseDirection.z >= 0

                var hitIndex: Int = -1
                var bestData: TriangleIntersection? = nil
                var localTHit = Float(tHit)

                var stack = Stack128()
                var toVisit = 0
                var current = 0

                while true {
                        let node = nodesPointer[current]
                        let (_, hitMask) = node.intersect8(
                                rdirX: rdirX, rdirY: rdirY, rdirZ: rdirZ,
                                orgRdirX: orgRdirX, orgRdirY: orgRdirY, orgRdirZ: orgRdirZ,
                                nearXIsMin: nearXIsMin, nearYIsMin: nearYIsMin, nearZIsMin: nearZIsMin,
                                tHit: localTHit)

                        var mask: UInt8 = 0
                        if hitMask[0] { mask |= 1 }
                        if hitMask[1] { mask |= 2 }
                        if hitMask[2] { mask |= 4 }
                        if hitMask[3] { mask |= 8 }
                        if hitMask[4] { mask |= 16 }
                        if hitMask[5] { mask |= 32 }
                        if hitMask[6] { mask |= 64 }
                        if hitMask[7] { mask |= 128 }

                        if mask != 0 {
                                // Embree-style: handle 1 hit case directly (most common)
                                let r0 = Int(mask.trailingZeroBitCount)
                                mask &= mask &- 1

                                if node.primitiveCounts[r0] > 0 { // leaf
                                        var currentData = TriangleIntersection()
                                        let count = Int(node.primitiveCounts[r0])
                                        let offset = Int(node.primitiveOffsets[r0])
                                        var k = 0
                                        while k < count {
                                                var realTHit = Real(localTHit)
                                                if scene.getIntersectionData(
                                                        primId: primIdsPointer[offset + k],
                                                        ray: ray,
                                                        tHit: &realTHit,
                                                        data: &currentData) {
                                                        localTHit = Float(realTHit)
                                                        hitIndex = offset + k
                                                        bestData = currentData
                                                }
                                                k += 1
                                        }
                                }

                                if mask == 0 {
                                        // 1 hit: if interior, continue directly (no stack push)
                                        if node.primitiveCounts[r0] <= 0 {
                                                let childIdx = Int(node.childNodes[r0])
                                                if childIdx >= 0 {
                                                        current = childIdx
                                                        continue
                                                }
                                        }
                                } else {
                                        // 2+ hits: push first child if interior, then process remaining
                                        if node.primitiveCounts[r0] <= 0 {
                                                let childIdx = Int(node.childNodes[r0])
                                                if childIdx >= 0 {
                                                        stack[toVisit] = childIdx
                                                        toVisit += 1
                                                }
                                        }

                                        // Process remaining hit children
                                        while mask != 0 {
                                                let ri = Int(mask.trailingZeroBitCount)
                                                mask &= mask &- 1

                                                if node.primitiveCounts[ri] > 0 { // leaf
                                                        var currentData = TriangleIntersection()
                                                        let count = Int(node.primitiveCounts[ri])
                                                        let offset = Int(node.primitiveOffsets[ri])
                                                        var k = 0
                                                        while k < count {
                                                                var realTHit = Real(localTHit)
                                                                if scene.getIntersectionData(
                                                                        primId: primIdsPointer[offset + k],
                                                                        ray: ray,
                                                                        tHit: &realTHit,
                                                                        data: &currentData) {
                                                                        localTHit = Float(realTHit)
                                                                        hitIndex = offset + k
                                                                        bestData = currentData
                                                                }
                                                                k += 1
                                                        }
                                                } else { // interior
                                                        let childIdx = Int(node.childNodes[ri])
                                                        if childIdx >= 0 {
                                                                stack[toVisit] = childIdx
                                                                toVisit += 1
                                                        }
                                                }
                                        }


                                }
                        }

                        if toVisit == 0 { break }
                        toVisit -= 1
                        current = stack[toVisit]
                }

                tHit = Real(localTHit)
                if let gdata = bestData, hitIndex >= 0 {
                        return scene.computeSurfaceInteraction(
                                primId: primIdsPointer[hitIndex],
                                data: gdata,
                                worldRay: ray)
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
