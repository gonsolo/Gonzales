struct Statistics {

        func report() {
                print("Statistics:")
                BoundingHierarchyBuilder.statistics()
                BoundingHierarchy.statistics()
                PerspectiveCamera.statistics()
                Triangle.statistics()
                Bounds3.statistics()
        }
}

var statistics = Statistics()
