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

        func intersect_lean(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitive? {
                var interaction = SurfaceInteraction()
                try intersect(ray: worldRay, tHit: &tHit, interaction: &interaction)
                if interaction.valid {
                        return .embreeCurve(self)
                } else {
                        return nil
                }
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                unimplemented()
        }

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX) throws -> SurfaceInteraction
        {
                unimplemented()
        }

        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        func objectBound() -> Bounds3f {
                var bounds = Bounds3f()
                for point in controlPoints {
                        bounds.add(point: point)
                }
                let width = max(widths.0, widths.1)
                bounds = expand(bounds: bounds, by: width)
                return bounds
        }

        func sample(u: TwoRandomVariables) -> (interaction: InteractionType, pdf: FloatX) {
                unimplemented()
        }

        func sample(ref: InteractionType, u: TwoRandomVariables) -> (InteractionType, FloatX) {
                unimplemented()
        }

        func probabilityDensityFor(
                samplingDirection direction: Vector,
                from interaction: InteractionType
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
) -> [ShapeType] {

        let curve = EmbreeCurve(
                objectToWorld: objectToWorld,
                controlPoints: controlPoints,
                widths: widths)
        let shape = ShapeType.embreeCurve(curve)
        return [shape]
}
