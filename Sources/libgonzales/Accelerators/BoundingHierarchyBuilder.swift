private enum SplitStrategy {
        case equal
        case middle
        case surfaceArea
}
private let splitStrategy = SplitStrategy.surfaceArea

enum BoundingHierarchyBuilderError: Error {
        case unknown
}

final class BVHBuildNode: @unchecked Sendable {
        var bounds: Bounds3f
        var left: BVHBuildNode?
        var right: BVHBuildNode?
        var splitAxis: Int
        var firstPrimOffset: Int
        var nPrimitives: Int

        init(offset: Int, nPrimitives: Int, bounds: Bounds3f) {
                self.firstPrimOffset = offset
                self.nPrimitives = nPrimitives
                self.bounds = bounds
                self.splitAxis = 0
        }

        init(axis: Int, left: BVHBuildNode, right: BVHBuildNode) {
                self.splitAxis = axis
                self.left = left
                self.right = right
                self.bounds = union(first: left.bounds, second: right.bounds)
                self.nPrimitives = 0
                self.firstPrimOffset = 0
        }
}

private struct CachedPrimitive {
        let index: Int
        let bound: Bounds3f
        let center: Point

        func centroid() -> Point {
                return 0.5 * bound.pMin + 0.5 * bound.pMax
        }
}

private struct UnsafePrimitiveBuffer: @unchecked Sendable {
        let buffer: UnsafeMutableBufferPointer<CachedPrimitive>
}

struct BVHSplitBucket {
        var count = 0
        var bounds = Bounds3f()
}

final class BoundingHierarchyBuilder: @unchecked Sendable {

        private let primitivesPerNode = 4

        private var totalNodes = 0
        private var bufferWrapper: UnsafePrimitiveBuffer?
        private var primitives: [IntersectablePrimitive]

        var bvh2Nodes = [BVH2Node]()

        internal init(scene: Scene, primitives: [IntersectablePrimitive]) async throws {
                self.primitives = primitives

                let pointer = UnsafeMutablePointer<CachedPrimitive>.allocate(capacity: primitives.count)
                let buffer = UnsafeMutableBufferPointer(start: pointer, count: primitives.count)

                let primitiveBufferWrapper = UnsafePrimitiveBuffer(buffer: buffer)
                await withTaskGroup(of: Void.self) { group in
                        let chunkSize = 32768
                        var startIndex = 0
                        while startIndex < primitives.count {
                                let endIndex = min(startIndex + chunkSize, primitives.count)
                                group.addTask { [startIndex, endIndex, scene] in
                                        for index in startIndex..<endIndex {
                                                let primitive = primitives[index]
                                                let bound = primitive.worldBound(scene: scene)
                                                primitiveBufferWrapper.buffer[index] = CachedPrimitive(
                                                        index: index, bound: bound, center: bound.center)
                                        }
                                }
                                startIndex += chunkSize
                        }
                        for await _ in group {}
                }

                self.bufferWrapper = UnsafePrimitiveBuffer(buffer: buffer)

                try await buildHierarchy()
        }

        deinit {
                if let wrapper = bufferWrapper {
                        wrapper.buffer.baseAddress?.deallocate()
                }
        }

        internal func getBoundingHierarchy() throws -> BoundingHierarchy {
                guard let wrapper = bufferWrapper else {
                        return BoundingHierarchy(primitives: [], bvh2Nodes: [])
                }
                let sortedPrimitives = wrapper.buffer.map { primitives[$0.index] }
                return BoundingHierarchy(primitives: sortedPrimitives, bvh2Nodes: bvh2Nodes)
        }

        private func buildHierarchy() async throws {
                guard let wrapper = bufferWrapper, wrapper.buffer.count > 0 else { return }
                let rootNode = try await Self.build(
                        range: 0..<wrapper.buffer.count, ptr: wrapper,
                        primitivesPerNode: self.primitivesPerNode)
                _ = flatten2(node: rootNode)
        }

        // --- BVH2 Flattening: depth-first linear layout ---
        // Left child is always at index + 1; right child index stored in .offset.
        // Each node stores its own AABB.

        private func flatten2(node: BVHBuildNode) -> Int {
                let myIndex = bvh2Nodes.count
                totalNodes += 1

                // Append placeholder — we'll fill it in below
                bvh2Nodes.append(BVH2Node())

                var n = BVH2Node()
                n.boundsMinX = Float(node.bounds.pMin.x)
                n.boundsMinY = Float(node.bounds.pMin.y)
                n.boundsMinZ = Float(node.bounds.pMin.z)
                n.boundsMaxX = Float(node.bounds.pMax.x)
                n.boundsMaxY = Float(node.bounds.pMax.y)
                n.boundsMaxZ = Float(node.bounds.pMax.z)

                if node.nPrimitives > 0 {
                        // Leaf
                        n.offset = Int32(node.firstPrimOffset)
                        n.count = Int32(node.nPrimitives)
                } else {
                        // Interior — left child is at myIndex + 1
                        if let left = node.left {
                                _ = flatten2(node: left)
                        }
                        // Right child gets the next available index
                        if let right = node.right {
                                let rightIndex = flatten2(node: right)
                                n.offset = Int32(rightIndex)
                        }
                        n.count = 0
                }

                bvh2Nodes[myIndex] = n
                return myIndex
        }

        // --- Static Helper functions ---

        private static func isSmaller(_ primitive: CachedPrimitive, _ pivot: Real, in dimension: Int) -> Bool
        {
                return primitive.center[dimension] < pivot
        }

        private static func isSmaller(_ first: CachedPrimitive, _ second: CachedPrimitive, in dimension: Int)
                -> Bool
        {
                return isSmaller(first, second.center[dimension], in: dimension)
        }

        private static func splitEqual(
                bounds _: Bounds3f, dimension: Int, range: Range<Int>, ptr: UnsafePrimitiveBuffer
        )
                -> (start: Int, middle: Int, end: Int)
        {
                ptr.buffer[range].sort(by: { isSmaller($0, $1, in: dimension) })
                let start = range.lowerBound
                let mid = start + range.count / 2
                let end = range.upperBound
                return (start, mid, end)
        }

        private static func splitMiddle(
                bounds: Bounds3f, dimension: Int, range: Range<Int>, ptr: UnsafePrimitiveBuffer
        )
                -> (start: Int, middle: Int, end: Int)
        {
                let pivot = (bounds.pMin[dimension] + bounds.pMax[dimension]) / 2
                let mid = ptr.buffer[range].partition(by: { isSmaller($0, pivot, in: dimension) })
                let start = range.lowerBound
                let end = range.upperBound
                guard mid != start && mid != end else {
                        return splitEqual(bounds: bounds, dimension: dimension, range: range, ptr: ptr)
                }
                return (start, mid, end)
        }

        private static func splitSurfaceAreaHeuristic(
                bounds: Bounds3f,
                centroidBounds: Bounds3f,
                dimension: Int,
                range: Range<Int>,
                ptr: UnsafePrimitiveBuffer,
                primitivesPerNode: Int
        ) -> (start: Int, middle: Int, end: Int, bounds: Bounds3f) {
                var start = 0
                var mid = 0
                var end = 0
                if range.count <= 2 {
                        mid = range.lowerBound + range.count / 2
                        ptr.buffer[range].sort(by: { isSmaller($0, $1, in: dimension) })
                } else {
                        let nBuckets = 12
                        var buckets = Array(repeating: BVHSplitBucket(), count: nBuckets)
                        for prim in ptr.buffer[range] {
                                let offset: Vector = centroidBounds.offset(point: prim.centroid())
                                var bucketIndex = Int(Float(nBuckets) * offset[dimension])
                                if bucketIndex == nBuckets { bucketIndex = nBuckets - 1 }
                                buckets[bucketIndex].count += 1
                                buckets[bucketIndex].bounds = union(
                                        first: buckets[bucketIndex].bounds, second: prim.bound)
                        }
                        let nSplits = nBuckets - 1
                        var costs = Array(repeating: Real(0.0), count: nSplits)
                        var countBelow = 0
                        var boundBelow = Bounds3f()
                        for index in 0..<nSplits {
                                boundBelow = union(first: boundBelow, second: buckets[index].bounds)
                                countBelow += buckets[index].count
                                costs[index] += Real(countBelow) * boundBelow.surfaceArea()
                        }
                        var countAbove = 0
                        var boundAbove = Bounds3f()
                        for index in (1...nSplits).reversed() {
                                boundAbove = union(first: boundAbove, second: buckets[index].bounds)
                                countAbove += buckets[index].count
                                costs[index - 1] += Real(countAbove) * boundAbove.surfaceArea()
                        }
                        var minCostSplitBucket = -1
                        var minCost = Real.infinity
                        for index in 0..<nSplits where costs[index] < minCost {
                                minCost = costs[index]
                                minCostSplitBucket = index
                        }
                        let leafCost = Real(range.count)
                        minCost = 1.0 / 2.0 + minCost / bounds.surfaceArea()

                        if range.count > primitivesPerNode || minCost < leafCost {
                                mid = ptr.buffer[range].partition(by: {
                                        let offsetPoint = centroidBounds.offset(point: $0.centroid())
                                        let offset = offsetPoint[dimension]
                                        var bucketIndex = Int(Real(nBuckets) * offset)
                                        if bucketIndex == nBuckets { bucketIndex = nBuckets - 1 }
                                        return bucketIndex > minCostSplitBucket
                                })
                        } else {
                                return (0, 0, 0, bounds)
                        }
                }
                start = range.lowerBound
                end = range.upperBound
                return (start, mid, end, Bounds3f())
        }

        private static func build(range: Range<Int>, ptr: UnsafePrimitiveBuffer, primitivesPerNode: Int)
                async throws -> BVHBuildNode
        {
                if range.isEmpty { return BVHBuildNode(offset: 0, nPrimitives: 0, bounds: Bounds3f()) }

                var bounds = Bounds3f()
                for prim in ptr.buffer[range] { bounds = union(first: bounds, second: prim.bound) }

                var centroidBounds = Bounds3f()
                for prim in ptr.buffer[range] {
                        centroidBounds = union(bound: centroidBounds, point: prim.center)
                }

                let dim = centroidBounds.maximumExtent()

                if bounds.surfaceArea() == 0 || range.count == 1
                        || centroidBounds.pMax[dim] == centroidBounds.pMin[dim]
                {
                        return BVHBuildNode(
                                offset: range.lowerBound, nPrimitives: range.count, bounds: bounds)
                }

                var start = 0
                var mid = 0
                var end = 0
                var sahLeafBounds = Bounds3f()

                switch splitStrategy {
                case .equal:
                        (start, mid, end) = splitEqual(
                                bounds: centroidBounds, dimension: dim, range: range, ptr: ptr)
                case .middle:
                        (start, mid, end) = splitMiddle(
                                bounds: centroidBounds, dimension: dim, range: range, ptr: ptr)
                case .surfaceArea:
                        (start, mid, end, sahLeafBounds) = splitSurfaceAreaHeuristic(
                                bounds: bounds, centroidBounds: centroidBounds, dimension: dim,
                                range: range, ptr: ptr, primitivesPerNode: primitivesPerNode)
                }

                if start == 0 && mid == 0 && end == 0 {
                        return BVHBuildNode(
                                offset: range.lowerBound, nPrimitives: range.count, bounds: sahLeafBounds)
                }

                if range.count > 100_000 {
                        let immutableStart = start
                        let immutableMid = mid
                        let immutableEnd = end
                        return try await withThrowingTaskGroup(of: BVHBuildNode.self) { group in
                                let immutablePtr = ptr
                                group.addTask {
                                        return try await build(
                                                range: immutableStart..<immutableMid, ptr: immutablePtr,
                                                primitivesPerNode: primitivesPerNode)
                                }
                                let rightNode = try await build(
                                        range: immutableMid..<immutableEnd, ptr: ptr,
                                        primitivesPerNode: primitivesPerNode)
                                let leftNode =
                                        try await group.next()
                                        ?? BVHBuildNode(offset: 0, nPrimitives: 0, bounds: Bounds3f())
                                return BVHBuildNode(axis: dim, left: leftNode, right: rightNode)
                        }
                } else {
                        let leftNode = try await build(
                                range: start..<mid, ptr: ptr, primitivesPerNode: primitivesPerNode)
                        let rightNode = try await build(
                                range: mid..<end, ptr: ptr, primitivesPerNode: primitivesPerNode)
                        return BVHBuildNode(axis: dim, left: leftNode, right: rightNode)
                }
        }
}
