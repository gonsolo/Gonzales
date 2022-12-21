private enum SplitStrategy {
        case equal
        case middle
        case surfaceArea
}
private let splitStrategy = SplitStrategy.surfaceArea

enum BoundingHierarchyBuilderError: Error {
        case unknown
}

final class BoundingHierarchyBuilder {

        internal init(primitives: [Boundable]) {
                self.cachedPrimitives = primitives.enumerated().map { index, primitive in
                        let bound = primitive.worldBound()
                        return CachedPrimitive(index: index, bound: bound, center: bound.center)
                }
                self.primitives = primitives
                buildHierarchy()
        }

        internal func getBoundingHierarchy() throws -> BoundingHierarchy {
                BoundingHierarchyBuilder.bhPrimitives += cachedPrimitives.count
                BoundingHierarchyBuilder.bhNodes += nodes.count
                let sortedPrimitives = try cachedPrimitives.map {
                        //primitives[$0.index] as! Intersectable
                        if let triangle = primitives[$0.index] as? Triangle {
                                return IntersectablePrimitive.triangle(triangle)
                        }
                        if let geometricPrimitive = primitives[$0.index] as? GeometricPrimitive {
                                return IntersectablePrimitive.geometricPrimitive(geometricPrimitive)
                        }
                        if let areaLight = primitives[$0.index] as? AreaLight {
                                return IntersectablePrimitive.areaLight(areaLight)
                        }
                        print("Unknown primitive \(primitives[$0.index])")
                        throw BoundingHierarchyBuilderError.unknown
                }
                return BoundingHierarchy(primitives: sortedPrimitives)
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

                func centroid() -> Point {
                        return 0.5 * bound.pMin + 0.5 * bound.pMax
                }
        }

        private func printNodes() {
                var i = 0
                for node in nodes {
                        print(i)
                        if node.count == 0 {
                                print("  interior")
                                print("  offset: ", node.offset)
                        } else {
                                print(" leaf")
                                print(" count: ", node.count)
                                print(" offset: ", node.offset)
                        }
                        i += 1
                }
        }

        private func buildHierarchy() {
                if cachedPrimitives.isEmpty { return }
                let _ = build(range: 0..<cachedPrimitives.count)
                //printNodes()
        }

        private func growNodes(counter: Int) {
                let missing = counter - nodes.count + 1
                if missing > 0 {
                        nodes += Array(repeating: Node(), count: missing)
                }
        }

        private func addLeafNode(
                offset: Int,
                bounds: Bounds3f,
                range: Range<Int>,
                counter: Int,
                dimension: Int
        ) {
                growNodes(counter: counter)
                assert(range.count > 0)
                nodes[counter] = Node(bounds: bounds, count: range.count, offset: offset, axis: 0)
                BoundingHierarchyBuilder.leafNodes += 1
                offsetCounter += range.count
                BoundingHierarchyBuilder.totalPrimitives += range.count
        }

        private func isSmaller(_ a: CachedPrimitive, _ pivot: FloatX, in dimension: Int) -> Bool {
                return a.center[dimension] < pivot
        }

        private func isSmaller(_ a: CachedPrimitive, _ b: CachedPrimitive, in dimension: Int)
                -> Bool
        {
                return isSmaller(a, b.center[dimension], in: dimension)
        }

        private func splitMiddle(bounds: Bounds3f, dimension: Int, range: Range<Int>)
                -> (start: Int, middle: Int, end: Int)
        {
                let pivot = (bounds.pMin[dimension] + bounds.pMax[dimension]) / 2
                let mid = cachedPrimitives[range].partition(by: {
                        isSmaller($0, pivot, in: dimension)
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
                cachedPrimitives[range].sort(by: { isSmaller($0, $1, in: dimension) })
                let start = range.first!
                let mid = start + range.count / 2
                let end = range.last! + 1
                return (start, mid, end)
        }

        struct BVHSplitBucket {
                var count = 0
                var bounds = Bounds3f()
        }

        private func splitSurfaceAreaHeuristic(
                bounds: Bounds3f,
                centroidBounds: Bounds3f,
                dimension: Int,
                range: Range<Int>,
                counter: Int
        )
                -> (start: Int, middle: Int, end: Int, bounds: Bounds3f)
        {
                var start = 0
                var mid = 0
                var end = 0
                if range.count <= 2 {
                        mid = range.first! + range.count / 2
                        cachedPrimitives[range].sort(by: { isSmaller($0, $1, in: dimension) })
                } else {
                        let nBuckets = 12
                        var buckets = Array(repeating: BVHSplitBucket(), count: nBuckets)
                        for prim in cachedPrimitives[range] {
                                let offset: Vector = centroidBounds.offset(point: prim.centroid())
                                var b = Int(Float(nBuckets) * offset[dimension])
                                if b == nBuckets {
                                        b = nBuckets - 1
                                }
                                assert(b >= 0)
                                assert(b < nBuckets)
                                buckets[b].count += 1
                                buckets[b].bounds = union(
                                        first: buckets[b].bounds,
                                        second: prim.bound)
                        }
                        let nSplits = nBuckets - 1
                        var costs = Array(repeating: FloatX(0.0), count: nSplits)
                        var countBelow = 0
                        var boundBelow = Bounds3f()
                        for i in 0..<nSplits {
                                boundBelow = union(first: boundBelow, second: buckets[i].bounds)
                                countBelow += buckets[i].count
                                costs[i] = costs[i] + FloatX(countBelow) * boundBelow.surfaceArea()
                        }
                        var countAbove = 0
                        var boundAbove = Bounds3f()
                        for i in (1...nSplits).reversed() {
                                boundAbove = union(first: boundAbove, second: buckets[i].bounds)
                                countAbove += buckets[i].count
                                costs[i - 1] =
                                        costs[i - 1] + FloatX(countAbove) * boundAbove.surfaceArea()
                        }
                        var minCostSplitBucket = -1
                        var minCost = FloatX.infinity
                        for i in 0..<nSplits {
                                if costs[i] < minCost {
                                        minCost = costs[i]
                                        minCostSplitBucket = i
                                }
                        }
                        let leafCost = FloatX(range.count)

                        minCost = 1.0 / 2.0 + minCost / bounds.surfaceArea()

                        if range.count > primitivesPerNode || minCost < leafCost {

                                mid = cachedPrimitives[range].partition(by: {
                                        let offsetPoint = centroidBounds.offset(
                                                point: $0.centroid())
                                        let offset = offsetPoint[dimension]
                                        var b = Int(FloatX(nBuckets) * offset)
                                        if b == nBuckets {
                                                b = nBuckets - 1
                                        }
                                        return b > minCostSplitBucket
                                })
                        } else {
                                addLeafNode(
                                        offset: offsetCounter,
                                        bounds: bounds,
                                        range: range,
                                        counter: counter,
                                        dimension: dimension)
                                return (0, 0, 0, bounds)
                        }
                }
                start = range.first!
                end = range.last! + 1
                return (start, mid, end, Bounds3f())
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
                let centroidBounds = cachedPrimitives[range].reduce(
                        Bounds3f(),
                        {
                                union(
                                        bound: $0,
                                        point: $1.center)
                        })
                let dim = centroidBounds.maximumExtent()

                if bounds.surfaceArea() == 0
                        || range.count == 1
                        || centroidBounds.pMax[dim] == centroidBounds.pMin[dim]
                {
                        addLeafNode(
                                offset: offsetCounter,
                                bounds: bounds,
                                range: range,
                                counter: counter,
                                dimension: dim)
                        return bounds
                }

                var start = 0
                var mid = 0
                var end = 0
                var blaBounds = Bounds3f()
                switch splitStrategy {
                case .equal:
                        (start, mid, end) = splitEqual(
                                bounds: centroidBounds,
                                dimension: dim,
                                range: range)
                case .middle:
                        (start, mid, end) = splitMiddle(
                                bounds: centroidBounds,
                                dimension: dim,
                                range: range)
                case .surfaceArea:
                        (start, mid, end, blaBounds) = splitSurfaceAreaHeuristic(
                                bounds: bounds,
                                centroidBounds: centroidBounds,
                                dimension: dim,
                                range: range,
                                counter: counter)
                }
                if start == 0 && mid == 0 && end == 0 {
                        return blaBounds
                }

                let leftBounds = build(range: start..<mid)
                let beforeRight = totalNodes
                let rightBounds = build(range: mid..<end)
                let combinedBounds = union(first: leftBounds, second: rightBounds)

                addInteriorNode(
                        counter: counter,
                        combinedBounds: combinedBounds,
                        dim: dim,
                        beforeRight: beforeRight)
                return combinedBounds
        }

        func addInteriorNode(counter: Int, combinedBounds: Bounds3f, dim: Int, beforeRight: Int) {
                growNodes(counter: counter)
                nodes[counter] = Node(bounds: combinedBounds, offset: beforeRight, axis: dim)
                BoundingHierarchyBuilder.interiorNodes += 1
        }

        private let primitivesPerNode = 4

        private static var interiorNodes = 0
        private static var leafNodes = 0
        private static var totalPrimitives = 0
        private static var callsToPartition = 0
        private static var bhNodes = 0
        private static var bhPrimitives = 0

        private var totalNodes = 0
        private var offsetCounter = 0
        private var cachedPrimitives: [CachedPrimitive]
        private var primitives: [Boundable]
}
