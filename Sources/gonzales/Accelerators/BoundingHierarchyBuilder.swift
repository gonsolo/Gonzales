final class BoundingHierarchyBuilder {

        init(primitives: [Boundable]) {
                self.nodes = []
                self.cachedPrimitives = primitives.enumerated().map { index, primitive in
                        let bound = primitive.worldBound()
                        return (index, bound, bound.center)
                }
                self.primitives = primitives
                buildHierarchy()
        }

        func buildHierarchy() {
                if cachedPrimitives.isEmpty { return }
                nodes = []
                let _ = build(range: 0..<cachedPrimitives.count)
        }

        func getBoundingHierarchy() -> BoundingHierarchy {
                BoundingHierarchyBuilder.bhPrimitives += cachedPrimitives.count
                BoundingHierarchyBuilder.bhNodes += nodes.count
                let sortedPrimitives = cachedPrimitives.map {
                        primitives[$0.0] as! Intersectable
                }
                return BoundingHierarchy(primitives: sortedPrimitives, nodes: nodes)
        }

        private func growNodes(counter: Int) {
                let missing = counter - nodes.count + 1
                if missing > 0 {
                        nodes += Array(repeating: Node(), count: missing)
                }
        }

        private func appendAndInit(
                offset: Int,
                bounds: Bounds3f,
                range: Range<Int>,
                counter: Int
        ) {
                growNodes(counter: counter)
                nodes[counter].bounds = bounds
                assert(range.count > 0)
                nodes[counter].offset = offset
                nodes[counter].count = range.count
                BoundingHierarchyBuilder.leafNodes += 1
                offsetCounter += range.count
                BoundingHierarchyBuilder.totalPrimitives += range.count
        }

        func build(range: Range<Int>) -> Bounds3f {
                let counter = totalNodes
                totalNodes += 1
                if range.isEmpty { return Bounds3f() }
                let bounds = cachedPrimitives[range].reduce(
                        Bounds3f(),
                        {
                                union(
                                        first: $0,
                                        second: $1.1)
                        })
                if range.count < BoundingHierarchyBuilder.primitivesPerNode {
                        appendAndInit(
                                offset: offsetCounter,
                                bounds: bounds,
                                range: range,
                                counter: counter)
                        return bounds
                }
                let centroidBounds = cachedPrimitives[range].reduce(
                        Bounds3f(),
                        {
                                union(
                                        bound: $0,
                                        point: $1.2)
                        })
                let dim = centroidBounds.maximumExtent()
                if centroidBounds.pMax[dim] == centroidBounds.pMin[dim] {
                        appendAndInit(
                                offset: offsetCounter,
                                bounds: bounds,
                                range: range,
                                counter: counter)
                        return bounds
                }
                let pivot = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2
                let mid = cachedPrimitives[range].partition(by: {
                        $0.2[dim] < pivot
                })
                let start = range.first!
                let end = range.last! + 1

                guard mid != start && mid != end else {
                        fatalError("Partition error: \(start) \(mid) \(end)!")
                }
                let leftBounds = build(range: start..<mid)
                let beforeRight = totalNodes
                let rightBounds = build(range: mid..<end)
                let combinedBounds = union(first: leftBounds, second: rightBounds)
                growNodes(counter: counter)
                nodes[counter].bounds = combinedBounds
                nodes[counter].axis = dim
                nodes[counter].count = 0
                nodes[counter].offset = beforeRight
                BoundingHierarchyBuilder.interiorNodes += 1
                return combinedBounds
        }

        static func statistics() {
                print("  BVH:")
                print("    Interior nodes:\t\t\t\t\t\t\t\(interiorNodes)")
                print("    Leaf nodes:\t\t\t\t\t\t\t\t\(leafNodes)")
                let ratio = String(format: " (%.2f)", Float(totalPrimitives) / Float(leafNodes))
                print("    Primitives per leaf node:\t\t\t\t\t\(totalPrimitives) /    \(leafNodes)\(ratio)")
        }

        static let primitivesPerNode = 1

        static var interiorNodes = 0
        static var leafNodes = 0
        static var totalPrimitives = 0
        static var callsToPartition = 0
        static var bhNodes = 0
        static var bhPrimitives = 0

        var totalNodes = 0
        var offsetCounter = 0
        var nodes: [Node]
        var cachedPrimitives: [(Int, Bounds3f, Point)]
        var primitives: [Boundable]
}
