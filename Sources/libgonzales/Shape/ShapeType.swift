import DevirtualizeMacro

enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)

        func getObjectToWorld(scene: Scene) -> Transform {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.getObjectToWorld(scene: scene)
                }
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.objectBound(scene: scene)
                }
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.worldBound(scene: scene)
                }
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) -> Bool {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func sample(samples: TwoRandomVariables, scene: Scene) -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.sample(samples: samples, scene: scene)
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, scene: Scene) -> (
                SurfaceInteraction, Real
        ) {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.sample(point: point, samples: samples, scene: scene)
                }
        }

        func area(scene: Scene) -> Area {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.area(scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: I
        ) -> Real {
                return #dispatchShapeTypeNoThrow(shape: self) { (typedShape: Triangle) in
                        typedShape.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: interaction)
                }
        }
}
