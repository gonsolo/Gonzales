import mojoKernel

/// Compact BVH2 node: 32 bytes = 1 cache line.
/// Each node stores its own AABB. Interior nodes have count == 0 and offset
/// points to the right child (left child is always at index + 1).
/// Leaf nodes have count > 0 and offset is the index into the primIds array.
struct BVH2Node {
        var boundsMinX: Float = .infinity
        var boundsMinY: Float = .infinity
        var boundsMinZ: Float = .infinity
        var boundsMaxX: Float = -.infinity
        var boundsMaxY: Float = -.infinity
        var boundsMaxZ: Float = -.infinity
        var offset: Int32 = 0
        var count: Int32 = 0
}
