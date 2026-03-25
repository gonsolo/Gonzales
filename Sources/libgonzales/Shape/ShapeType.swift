import DevirtualizeMacro

enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)

        func getObjectToWorld(scene: Scene) -> Transform {
                return #dispatchShapeTypeNoThrow(shape: self) { (p: Triangle) in
                        p.getObjectToWorld(scene: scene)
                }
        }

        func objectBound(scene: Scene) throws -> Bounds3f {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.objectBound(scene: scene)
                }
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.worldBound(scene: scene)
                }
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func sample(samples: TwoRandomVariables, scene: Scene) throws -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.sample(samples: samples, scene: scene)
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, scene: Scene) throws -> (
                SurfaceInteraction, Real
        ) {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.sample(point: point, samples: samples, scene: scene)
                }
        }

        func area(scene: Scene) throws -> Area {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.area(scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: I
        ) throws -> Real {
                return try #dispatchShapeType(shape: self) { (p: Triangle) in
                        try p.probabilityDensityFor(scene: scene, samplingDirection: direction, from: interaction)
                }
        }
}
