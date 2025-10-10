struct Statistics {

        @MainActor
        func report() {
                print("Statistics:")
                BoundingHierarchyBuilder.statistics()
                BoundingHierarchy<IntersectablePrimitive>.statistics()
                PerspectiveCamera.statistics()
                Triangle.statistics()
                Bounds3.statistics()
        }
}

@MainActor
var statistics = Statistics()
