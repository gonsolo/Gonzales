final class BoundingHierarchyBuilder {

        internal init(primitives: [Boundable]) {
                self.nodes = []
                self.cachedPrimitives = primitives.enumerated().map { index, primitive in
                        let bound = primitive.worldBound()
                        return CachedPrimitive(index: index, bound: bound, center: bound.center)
                }
                self.primitives = primitives
                buildHierarchy()
        }

        internal func getBoundingHierarchy() -> BoundingHierarchy {
                BoundingHierarchyBuilder.bhPrimitives += cachedPrimitives.count
                BoundingHierarchyBuilder.bhNodes += nodes.count
                let sortedPrimitives = cachedPrimitives.map {
                        primitives[$0.index] as! Intersectable
                }
                return BoundingHierarchy(primitives: sortedPrimitives, nodes: nodes)
        }

        internal static func statistics() {
                print("  BVH:")
                print("    Interior nodes:\t\t\t\t\t\t\t\(interiorNodes)")
                print("    Leaf nodes:\t\t\t\t\t\t\t\t\(leafNodes)")
                let ratio = String(format: " (%.2f)", Float(totalPrimitives) / Float(leafNodes))
                print("    Primitives per leaf node:\t\t\t\t\t", terminator: "")
                print("\(totalPrimitives) /    \(leafNodes)\(ratio)")
        }

        private struct CachedPrimitive {
                let index: Int
                let bound: Bounds3f
                let center: Point
        }

        private func buildHierarchy() {
                if cachedPrimitives.isEmpty { return }
                nodes = []
                let _ = build(range: 0..<cachedPrimitives.count)
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

        private func splitMiddle(bounds: Bounds3f, dimension: Int, range: Range<Int>)
                -> (start: Int, middle: Int, end: Int)
        {
                let pivot = (bounds.pMin[dimension] + bounds.pMax[dimension]) / 2
                let mid = cachedPrimitives[range].partition(by: {
                        $0.center[dimension] < pivot
                })
                let start = range.first!
                let end = range.last! + 1
                guard mid != start && mid != end else {
                        return splitEqual(bounds: bounds, dimension: dimension, range: range)
                }
                return (start, mid, end)
        }

        private func splitEqual(bounds: Bounds3f, dimension: Int, range: Range<Int>)
                -> (start: Int, middle: Int, end: Int)
        {
                // There is no nth_element so let's sort for now
                cachedPrimitives[range].sort(by: { $0.center[dimension] < $1.center[dimension] })
                let start = range.first!
                let mid = start + cachedPrimitives[range].count / 2
                let end = range.last! + 1
                return (start, mid, end)
        }

        private func splitSurfaceAreaHeuristic(bounds: Bounds3f, dimension: Int, range: Range<Int>)
                -> (start: Int, middle: Int, end: Int)
        {
                // TODO
                return (0, 1, 2)
        }

        private func build(range: Range<Int>) -> Bounds3f {
                let counter = totalNodes
                totalNodes += 1
                if range.isEmpty { return Bounds3f() }
                let bounds = cachedPrimitives[range].reduce(
                        Bounds3f(),
                        {
                                union(first: $0, second: $1.bound)
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
                                        point: $1.center)
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

                let (start, mid, end) = splitMiddle(
                        bounds: centroidBounds,
                        dimension: dim,
                        range: range)

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

        private static let primitivesPerNode = 1

        private static var interiorNodes = 0
        private static var leafNodes = 0
        private static var totalPrimitives = 0
        private static var callsToPartition = 0
        private static var bhNodes = 0
        private static var bhPrimitives = 0

        private var totalNodes = 0
        private var offsetCounter = 0
        private var nodes: [Node]
        private var cachedPrimitives: [CachedPrimitive]
        private var primitives: [Boundable]
}
