// BoundingHierarchy.swift

struct BoundingHierarchy: Boundable, Intersectable, Sendable {

        init(primitives: [IntersectablePrimitive], nodes: [BoundingHierarchyNode]) {
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
                                        id1: triangle.meshIndex, id2: triangle.triangleIndex, type: .triangle)
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

        // --- 1. Private Traversal Protocol (The Leaf Logic Interface) ---
        private protocol LeafProcessor {
                mutating func processLeaf(
                        scene: Scene, hierarchy: BoundingHierarchy, node: BoundingHierarchyNode, ray: Ray)
                var tHit: Real { get set }
        }

        // --- 2. Private Traversal Helper (Generic Shared Logic) ---
        private func traverseHierarchy<P: LeafProcessor>(
                scene: Scene,
                ray: Ray,
                processor: inout P
        ) {
                var toVisit = 0
                var current = 0
                // Use a fixed-size array/tuple for performance
                var nodesToVisit: [32 of Int] = .init(repeating: 0)

                if nodes.isEmpty { return }

                while true {
                        let node = nodes[current]

                        // 1. Check intersection with the bounding box
                        if node.bounds.intersects(ray: ray, tHit: processor.tHit) {

                                if node.count > 0 {  // leaf node

                                        // 2. Execute the leaf-specific logic via the protocol method
                                        processor.processLeaf(
                                                scene: scene, hierarchy: self, node: node, ray: ray)

                                        // 3. Move to the next node from the stack (if any)
                                        if toVisit == 0 { break }
                                        toVisit -= 1
                                        current = nodesToVisit[toVisit]

                                } else {  // interior node
                                        // 4. Determine child traversal order (Closest child first)
                                        let firstChildIndex: Int
                                        let secondChildIndex: Int

                                        if ray.direction[node.axis] < 0 {
                                                firstChildIndex = node.offset
                                                secondChildIndex = current + 1
                                        } else {
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

        private struct OcclusionProcessor: LeafProcessor {
                var intersected: Bool = false
                var tHit: Real

                mutating func processLeaf(
                        scene: Scene, hierarchy: BoundingHierarchy, node: BoundingHierarchyNode, ray: Ray
                ) {
                        for index in 0..<node.count {
                                intersected =
                                        intersected
                                        || scene.intersect(
                                                primId: hierarchy.primIds[node.offset + index],
                                                ray: ray,
                                                tHit: &tHit)
                        }
                }
        }

        // --- Public Intersect (Occlusion Query) ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Bool {
                var processor = OcclusionProcessor(tHit: tHit)
                traverseHierarchy(scene: scene, ray: ray, processor: &processor)
                tHit = processor.tHit
                return processor.intersected
        }

        // --- 4. Interaction Leaf Logic (Value Type) ---
        private struct InteractionProcessor: LeafProcessor {
                var index: Int = 0
                var gdata: TriangleIntersection?
                var tHit: Real

                mutating func processLeaf(
                        scene: Scene, hierarchy: BoundingHierarchy, node: BoundingHierarchyNode, ray: Ray
                ) {
                        var currentData = TriangleIntersection()
                        for index in 0..<node.count {
                                let intersectionFound = scene.getIntersectionData(
                                        primId: hierarchy.primIds[node.offset + index],
                                        ray: ray,
                                        tHit: &tHit,
                                        data: &currentData)

                                if intersectionFound {
                                        self.index = node.offset + index
                                        gdata = currentData
                                }
                        }
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                var processor = InteractionProcessor(tHit: tHit)
                traverseHierarchy(scene: scene, ray: ray, processor: &processor)

                tHit = processor.tHit
                var interaction: SurfaceInteraction?
                if let gdata = processor.gdata {
                        interaction = scene.computeSurfaceInteraction(
                                primId: primIds[processor.index],
                                data: gdata,
                                worldRay: ray)
                }
                return interaction
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
        var primIds: [PrimId]
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
