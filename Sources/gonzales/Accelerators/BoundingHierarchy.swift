struct BoundingHierarchy: Boundable, Intersectable, Sendable {

        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitive? {
                var toVisit = 0
                var current = 0
                var nodesToVisit: [32 of Int] = .init(repeating: 0)
                var nodesVisited = 0
                var candidate: IntersectablePrimitive? = nil

                if nodes.isEmpty { return nil }
                while true {
                        nodesVisited += 1
                        let node = nodes[current]
                        if node.bounds.intersects(ray: ray, tHit: tHit) {
                                if node.count > 0 {  // leaf
                                        for i in 0..<node.count {
                                                let primitive = primitives[node.offset + i]
                                                if try primitive.intersect_lean(ray: ray, tHit: &tHit) != nil {
                                                        candidate = primitive
                                                }
                                        }
                                        if toVisit == 0 { break }
                                        toVisit -= 1
                                        current = nodesToVisit[toVisit]
                                } else {  // interior
                                        if ray.direction[node.axis] < 0 {
                                                nodesToVisit[toVisit] = current + 1
                                                current = node.offset
                                        } else {
                                                nodesToVisit[toVisit] = node.offset
                                                current = current + 1
                                        }
                                        toVisit += 1
                                }
                        } else {
                                if toVisit == 0 { break }
                                toVisit -= 1
                                current = nodesToVisit[toVisit]
                        }
                }
                return candidate
        }

        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction
        //) throws {
        //        let candidate = try intersect_lean(ray: ray, tHit: &tHit)
        //        try candidate?.intersect(
        //                ray: ray,
        //                tHit: &tHit,
        //                interaction: &interaction)
        //}

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction {
                let candidate = try intersect_lean(ray: ray, tHit: &tHit)
                if let candidate {
                        return try candidate.computeInteraction(ray: ray, tHit: &tHit)
                } else { 
                        return  SurfaceInteraction()
                }
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
                print("    Nodes visited:\t\t\t\t\t\t\t\(boundingHierarchyNodesVisited)")
        }

        let primitives: [IntersectablePrimitive]
        var nodes = [BoundingHierarchyNode]()

        @MainActor
        static var boundingHierarchyNodesVisited = 0
}
