/// A type that can be sampled.
///
/// For example, to render area lights a renderer typically chooses a point
/// on the area light and calculates the light from there. If many such
/// points are chosen and the light calculated, the average approximates
/// the analytical solution.

protocol Sampleable {

        func sample(u: TwoRandomVariables) -> (interaction: any Interaction, pdf: FloatX)

        func sample(ref: any Interaction, u: TwoRandomVariables) -> (any Interaction, FloatX)

        func probabilityDensityFor(
                samplingDirection direction: Vector,
                from interaction: any Interaction
        )
                throws -> FloatX

        func area() -> FloatX
}

extension Sampleable {

        func sample(ref: any Interaction, u: TwoRandomVariables) -> (any Interaction, FloatX) {
                var (intr, pdf) = sample(u: u)
                let wi: Vector = normalized(intr.position - ref.position)
                let squaredDistance = distanceSquared(ref.position, intr.position)
                let angle = absDot(Vector(normal: intr.normal), -wi)
                pdf *= squaredDistance / angle
                return (intr, pdf)
        }
}

extension Sampleable where Self: Intersectable {

        func probabilityDensityFor(
                samplingDirection direction: Vector,
                from: any Interaction
        )
                throws -> FloatX
        {
                let ray = from.spawnRay(inDirection: direction)
                var tHit: FloatX = 0.0
                var interaction = SurfaceInteraction()
                try intersect(ray: ray, tHit: &tHit, interaction: &interaction)
                if !interaction.valid {
                        return 0
                }
                let squaredDistance = distanceSquared(from.position, interaction.position)
                let angle = absDot(interaction.normal, -direction)
                let angleTimesArea = angle * area()
                let density = squaredDistance / angleTimesArea
                if density.isInfinite {
                        return 0
                }
                return density
        }
}
