// BoundingHierarchy.swift

final class BoundingHierarchy: Boundable, Intersectable, Sendable {

        init(primitives: [IntersectablePrimitive], nodes: [BoundingHierarchyNode]) {
                self.nodes = nodes
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
                self.primIds = ids
        }

        // --- Public Intersect (Occlusion Query) ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Bool {
                if nodes.isEmpty { return false }

                var isOccluded = false
                var localTHit = tHit

                nodes.withUnsafeBufferPointer { nodesBuffer in
                        primIds.withUnsafeBufferPointer { primsBuffer in
                                var toVisit = 0
                                var current = 0
                                var nodesToVisit: [32 of Int] = .init(repeating: 0)

                                while true {
                                        let node = nodesBuffer[current]

                                        if node.bounds.intersects(ray: ray, tHit: localTHit) {
                                                if node.count > 0 {  // leaf node
                                                        for index in 0..<node.count {
                                                                if scene.intersect(
                                                                        primId: primsBuffer[node.offset + index],
                                                                        ray: ray,
                                                                        tHit: &localTHit) {
                                                                        isOccluded = true
                                                                        break
                                                                }
                                                        }
                                                        if isOccluded { break }

                                                        if toVisit == 0 { break }
                                                        toVisit -= 1
                                                        current = nodesToVisit[toVisit]

                                                } else {  // interior node
                                                        let firstChildIndex: Int
                                                        let secondChildIndex: Int
                                                        if ray.direction[node.axis] < 0 {
                                                                firstChildIndex = node.offset
                                                                secondChildIndex = current + 1
                                                        } else {
                                                                firstChildIndex = current + 1
                                                                secondChildIndex = node.offset
                                                        }

                                                        nodesToVisit[toVisit] = secondChildIndex
                                                        toVisit += 1
                                                        current = firstChildIndex
                                                }
                                        } else {
                                                if toVisit == 0 { break }
                                                toVisit -= 1
                                                current = nodesToVisit[toVisit]
                                        }
                                }
                        }
                }
                
                tHit = localTHit
                return isOccluded
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if nodes.isEmpty { return nil }

                var hitIndex: Int = -1
                var bestData: TriangleIntersection? = nil
                var localTHit = tHit

                nodes.withUnsafeBufferPointer { nodesBuffer in
                        primIds.withUnsafeBufferPointer { primsBuffer in
                                var toVisit = 0
                                var current = 0
                                var nodesToVisit: [32 of Int] = .init(repeating: 0)

                                while true {
                                        let node = nodesBuffer[current]

                                        if node.bounds.intersects(ray: ray, tHit: localTHit) {
                                                if node.count > 0 {  // leaf node
                                                        var currentData = TriangleIntersection()
                                                        for index in 0..<node.count {
                                                                let intersectionFound = scene.getIntersectionData(
                                                                        primId: primsBuffer[node.offset + index],
                                                                        ray: ray,
                                                                        tHit: &localTHit,
                                                                        data: &currentData)

                                                                if intersectionFound {
                                                                        hitIndex = node.offset + index
                                                                        bestData = currentData
                                                                }
                                                        }

                                                        if toVisit == 0 { break }
                                                        toVisit -= 1
                                                        current = nodesToVisit[toVisit]

                                                } else {  // interior node
                                                        let firstChildIndex: Int
                                                        let secondChildIndex: Int
                                                        if ray.direction[node.axis] < 0 {
                                                                firstChildIndex = node.offset
                                                                secondChildIndex = current + 1
                                                        } else {
                                                                firstChildIndex = current + 1
                                                                secondChildIndex = node.offset
                                                        }

                                                        nodesToVisit[toVisit] = secondChildIndex
                                                        toVisit += 1
                                                        current = firstChildIndex
                                                }
                                        } else {
                                                if toVisit == 0 { break }
                                                toVisit -= 1
                                                current = nodesToVisit[toVisit]
                                        }
                                }
                        }
                }

                tHit = localTHit
                if let gdata = bestData, hitIndex >= 0 {
                        return scene.computeSurfaceInteraction(
                                primId: primIds[hitIndex],
                                data: gdata,
                                worldRay: ray)
                }
                return nil
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return worldBound(scene: scene)
        }

        func worldBound(scene _: Scene) -> Bounds3f {
                if nodes.isEmpty {
                        return Bounds3f()
                } else {
                        return nodes[0].bounds
                }
        }

        let nodes: [BoundingHierarchyNode]
        let primIds: [PrimId]
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
