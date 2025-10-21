struct BoundingHierarchy: Boundable, Intersectable, Sendable {

        init(primitives: [IntersectablePrimitive], nodes: [BoundingHierarchyNode]) {
                self.primitives = primitives
                self.nodes = nodes
                self.primIds = []
                for primitive in primitives {
                        switch primitive {
                        case .geometricPrimitive(let geometricPrimitive):
                                let primId = PrimId(
                                        id1: geometricPrimitive.idx, id2: -1, type: .geometricPrimitive)
                                self.primIds.append(primId)
                        case .triangle(let triangle):
                                let primId = PrimId(
                                        id1: triangle.meshIndex, id2: triangle.idx, type: .triangle)
                                self.primIds.append(primId)
                        case .transformedPrimitive(let transformedPrimitive):
                                let primId = PrimId(
                                        id1: transformedPrimitive.idx, id2: -1, type: .transformedPrimitive)
                                self.primIds.append(primId)
                        case .areaLight(let areaLight):
                                let primId = PrimId(id1: areaLight.idx, id2: -1, type: .areaLight)
                                self.primIds.append(primId)
                        }
                }
        }

        // --- Private Traversal Logic ---
        // The traversal function accepts a closure 'onLeaf' to execute when a leaf node is reached.
        private func traverseHierarchy(
                ray: Ray,
                tHit: FloatX,
                onLeaf: (_ node: BoundingHierarchyNode) throws -> Void
        ) rethrows {
                var toVisit = 0
                var current = 0
                var nodesToVisit: [32 of Int] = .init(repeating: 0)
                var nodesVisited = 0

                if nodes.isEmpty { return }

                while true {
                        nodesVisited += 1
                        let node = nodes[current]

                        // 1. Check intersection with the bounding box
                        if node.bounds.intersects(ray: ray, tHit: tHit) {

                                if node.count > 0 {  // leaf node

                                        // 2. Execute the leaf-specific logic provided by the caller
                                        try onLeaf(node)

                                        // 3. Move to the next node from the stack (if any)
                                        if toVisit == 0 { break }
                                        toVisit -= 1
                                        current = nodesToVisit[toVisit]

                                } else {  // interior node

                                        // 4. Determine child traversal order (closest child first)
                                        // The children are assumed to be current + 1 and node.offset.
                                        let firstChildIndex: Int
                                        let secondChildIndex: Int

                                        if ray.direction[node.axis] < 0 {
                                                // Negative direction: near child is at node.offset, far child is at current + 1
                                                firstChildIndex = node.offset
                                                secondChildIndex = current + 1
                                        } else {
                                                // Positive direction: near child is at current + 1, far child is at node.offset
                                                firstChildIndex = current + 1
                                                secondChildIndex = node.offset
                                        }

                                        // 5. Push the farther child onto the stack
                                        nodesToVisit[toVisit] = secondChildIndex
                                        toVisit += 1

                                        // 6. Set the current node to the nearer child
                                        current = firstChildIndex
                                }
                        } else {
                                // Missed bounding box, move to the next node from the stack
                                if toVisit == 0 { break }
                                toVisit -= 1
                                current = nodesToVisit[toVisit]
                        }
                }
        }
        // ------------------------------------

        // --- Public Intersect (Occlusion Query) ---
        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                var intersected = false

                try traverseHierarchy(ray: ray, tHit: tHit) { node in
                        for i in 0..<node.count {
                                //intersected =
                                //        try intersected
                                //        || primitives[node.offset + i].intersect(
                                //                ray: ray,
                                //                tHit: &tHit
                                //        )

                                intersected =
                                        try intersected
                                        || globalScene!.intersect(
                                                primId: primIds[node.offset + i],
                                                ray: ray,
                                                tHit: &tHit)
                        }
                }
                return intersected
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                var gi = 0
                var gnode = BoundingHierarchyNode()
                var gdata: IntersectablePrimitiveIntersection = .triangle(nil)

                try traverseHierarchy(ray: ray, tHit: tHit) { node in
                        for i in 0..<node.count {
                                //let data = try primitives[node.offset + i].getIntersectionData(
                                //        ray: ray, tHit: &tHit)

                                let data = try globalScene!.getIntersectionData(
                                        primId: primIds[node.offset + i],
                                        ray: ray,
                                        tHit: &tHit)

                                switch data {
                                case .triangle(let triangle):
                                        if triangle != nil {
                                                gi = i
                                                gnode = node
                                                gdata = data
                                        }
                                }
                        }
                }
                //primitives[gnode.offset + gi].computeSurfaceInteraction(
                //        data: gdata,
                //        worldRay: ray,
                //        interaction: &interaction)

                try globalScene!.computeSurfaceInteraction(
                        primId: primIds[gnode.offset + gi],
                        data: gdata,
                        worldRay: ray,
                        interaction: &interaction)

        }

        @MainActor
        func objectBound() -> Bounds3f {
                return worldBound()
        }

        @MainActor
        func worldBound() -> Bounds3f {
                if nodes.isEmpty {
                        return Bounds3f()
                } else {
                        return nodes[0].bounds
                }
        }

        @MainActor
        static func statistics() {
                //print("    Nodes visited:\t\t\t\t\t\t\t\(boundingHierarchyNodesVisited)")
        }

        let primitives: [IntersectablePrimitive]
        let nodes: [BoundingHierarchyNode]

        var primIds: [PrimId]
}

enum PrimType: UInt8 {
        case triangle
        case geometricPrimitive
        case transformedPrimitive
        case areaLight
}

struct PrimId {
        let id1: Int
        let id2: Int
        let type: PrimType
}
