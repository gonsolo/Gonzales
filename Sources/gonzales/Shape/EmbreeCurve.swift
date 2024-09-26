final class EmbreeCurve: Shape {

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

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                unimplemented()
        }

        @MainActor
        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        @MainActor
        func objectBound() -> Bounds3f {
                var bounds = Bounds3f()
                for point in controlPoints {
                        bounds.add(point: point)
                }
                let width = max(widths.0, widths.1)
                bounds = expand(bounds: bounds, by: width)
                return bounds
        }

        func sample(u: TwoRandomVariables) -> (interaction: Interaction, pdf: FloatX) {
                unimplemented()
        }

        func sample(ref: Interaction, u: TwoRandomVariables) -> (Interaction, FloatX) {
                unimplemented()
        }

        @MainActor
        func probabilityDensityFor(
                samplingDirection direction: Vector,
                from interaction: Interaction
        ) throws -> FloatX {
                unimplemented()
        }

        func area() -> FloatX {
                unimplemented()
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
) -> [Shape] {

        let curve = EmbreeCurve(
                objectToWorld: objectToWorld,
                controlPoints: controlPoints,
                widths: widths)
        var curves = [Shape]()
        curves.append(curve)
        return curves
}
