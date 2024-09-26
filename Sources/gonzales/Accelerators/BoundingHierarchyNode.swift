struct BoundingHierarchyNode {

        @MainActor
        init(bounds: Bounds3f, count: Int = 0, offset: Int = 0, axis: Int = 0) {
                self.bounds = bounds
                self.count = count
                self.offset = offset
                self.axis = axis
        }

        let bounds: Bounds3f
        let count: Int
        let offset: Int
        let axis: Int
}
