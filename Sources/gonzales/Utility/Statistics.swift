struct Statistics {

        func report() {
                print("Statistics:")
                BoundingHierarchyBuilder.statistics()
                BoundingHierarchy.statistics()
                PerspectiveCamera.statistics()
                Scene.statistics()
                Triangle.statistics()
                //Curve.statistics()
                Bounds3.statistics()
        }
}

var statistics = Statistics()
