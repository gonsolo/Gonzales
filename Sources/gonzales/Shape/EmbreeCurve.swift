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
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX,
                interaction _: inout SurfaceInteraction
        ) throws {
                unimplemented()
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene _: Scene) -> Bounds3f {
                var bounds = Bounds3f()
                for point in controlPoints {
                        bounds.add(point: point)
                }
                let width = max(widths.0, widths.1)
                bounds = expand(bounds: bounds, by: width)
                return bounds
        }

        func sample<I: Interaction>(samples _: TwoRandomVariables, scene _: Scene) -> (
                interaction: I, pdf: FloatX
        ) {
                unimplemented()
        }

        func sample(ref _: any Interaction, u _: TwoRandomVariables) -> (any Interaction, FloatX) {
                unimplemented()
        }

        func probabilityDensityFor(
                samplingDirection _: Vector,
                from _: any Interaction
        ) throws -> FloatX {
                unimplemented()
        }

        func area(scene _: Scene) -> FloatX {
                unimplemented()
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
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
