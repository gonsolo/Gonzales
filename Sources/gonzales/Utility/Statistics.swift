struct Statistics {

        func report() {
                print("Statistics:")
                BoundingHierarchyBuilder.statistics()
                BoundingHierarchy.statistics()
                renderer?.camera.statistics()
                scene.statistics()
                Triangle.statistics()
                //Curve.statistics()
                Bounds3.statistics()
        }
}

var statistics = Statistics()
