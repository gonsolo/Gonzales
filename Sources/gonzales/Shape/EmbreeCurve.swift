struct EmbreeCurve: Shape {

        @MainActor
        init(
                objectToWorld: Transform,
                controlPoints: [Point],
                widths: (Float, Float)
        ) {
                self.objectToWorld = objectToWorld
                self.controlPoints = controlPoints
                self.widths = widths
                numberOfCurves += 1
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                unimplemented()
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene: Scene) -> Bounds3f {
                var bounds = Bounds3f()
                for point in controlPoints {
                        bounds.add(point: point)
                }
                let width = max(widths.0, widths.1)
                bounds = expand(bounds: bounds, by: width)
                return bounds
        }

        func sample<I: Interaction>(u: TwoRandomVariables, scene: Scene) -> (interaction: I, pdf: FloatX) {
                unimplemented()
        }

        func sample(ref: any Interaction, u: TwoRandomVariables) -> (any Interaction, FloatX) {
                unimplemented()
        }

        func probabilityDensityFor(
                samplingDirection direction: Vector,
                from interaction: any Interaction
        ) throws -> FloatX {
                unimplemented()
        }

        func area(scene: Scene) -> FloatX {
                unimplemented()
        }

        func getObjectToWorld(scene: Scene) -> Transform {
                return objectToWorld
        }

        let objectToWorld: Transform
        let controlPoints: [Point]
        let widths: (Float, Float)
}

@MainActor
func createEmbreeCurveShape(
        controlPoints: [Point],
        widths: (Float, Float),
        objectToWorld: Transform
) -> [ShapeType] {

        let curve = EmbreeCurve(
                objectToWorld: objectToWorld,
                controlPoints: controlPoints,
                widths: widths)
        let shape = ShapeType.embreeCurve(curve)
        return [shape]
}
