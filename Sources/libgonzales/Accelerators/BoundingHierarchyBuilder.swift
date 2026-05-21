import mojoKernel

enum BoundingHierarchyBuilderError: Error {
        case unknown
}

private struct CachedPrimitive {
        let index: Int
        let bound: Bounds3f
        let center: Point
}

private struct UnsafePrimitiveBuffer: @unchecked Sendable {
        let buffer: UnsafeMutableBufferPointer<CachedPrimitive>
}

final class BoundingHierarchyBuilder: @unchecked Sendable {

        private var bufferWrapper: UnsafePrimitiveBuffer?
        private var primitives: [IntersectablePrimitive]
        private var primOrder = [Int]()

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

                try buildHierarchy()
        }

        deinit {
                if let wrapper = bufferWrapper {
                        wrapper.buffer.baseAddress?.deallocate()
                }
        }

        internal func getBoundingHierarchy() throws -> BoundingHierarchy {
                guard bufferWrapper != nil else {
                        return BoundingHierarchy(primitives: [], bvh2Nodes: [])
                }
                let sortedPrimitives = primOrder.map { primitives[$0] }
                return BoundingHierarchy(primitives: sortedPrimitives, bvh2Nodes: bvh2Nodes)
        }

        // BVH2 construction is performed in Mojo (mojo_build_bvh2). We feed it the
        // per-primitive AABBs computed above and receive back the depth-first node
        // array plus the primitive permutation (leaf order).
        private func buildHierarchy() throws {
                guard let wrapper = bufferWrapper, wrapper.buffer.count > 0 else { return }
                let n = wrapper.buffer.count

                var primBounds = [Float](repeating: 0, count: n * 6)
                for i in 0..<n {
                        let b = wrapper.buffer[i].bound
                        primBounds[i * 6 + 0] = Float(b.pMin.x)
                        primBounds[i * 6 + 1] = Float(b.pMin.y)
                        primBounds[i * 6 + 2] = Float(b.pMin.z)
                        primBounds[i * 6 + 3] = Float(b.pMax.x)
                        primBounds[i * 6 + 4] = Float(b.pMax.y)
                        primBounds[i * 6 + 5] = Float(b.pMax.z)
                }

                var outNodes = [BVH2Node](repeating: BVH2Node(), count: max(2 * n, 1))
                var outOrder = [Int32](repeating: 0, count: n)

                // The C header's `struct BVH2Node` and Swift's BVH2Node are layout-identical
                // but distinct types; rebind the buffer to bridge them.
                let nodeCount: Int32 = primBounds.withUnsafeBufferPointer { boundsPtr in
                        outNodes.withUnsafeMutableBufferPointer { nodesPtr in
                                outOrder.withUnsafeMutableBufferPointer { orderPtr in
                                        nodesPtr.baseAddress!.withMemoryRebound(
                                                to: mojoKernel.BVH2Node.self, capacity: nodesPtr.count
                                        ) { mNodes in
                                                mojo_build_bvh2(
                                                        boundsPtr.baseAddress, Int32(n),
                                                        mNodes, orderPtr.baseAddress)
                                        }
                                }
                        }
                }

                bvh2Nodes = Array(outNodes[0..<Int(nodeCount)])
                primOrder = outOrder.map { Int($0) }
        }
}
