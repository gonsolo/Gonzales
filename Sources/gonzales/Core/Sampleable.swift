/// A type that can be sampled.
///
/// For example, to render area lights a renderer typically chooses a point
/// on the area light and calculates the light from there. If many such
/// points are chosen and the light calculated, the average approximates
/// the analytical solution.

protocol Sampleable {

        func sample(u: TwoRandomVariables) -> (interaction: SurfaceInteraction, pdf: FloatX)

        func sample(point: Point, u: TwoRandomVariables) -> (SurfaceInteraction, FloatX)

        func probabilityDensityFor(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: SurfaceInteraction
        ) throws -> FloatX

        func area() -> FloatX
}

extension Sampleable {

        func sample(point: Point, u: TwoRandomVariables) -> (SurfaceInteraction, FloatX) {
                var (intr, pdf) = sample(u: u)
                let wi: Vector = normalized(intr.position - point)
                let squaredDistance = distanceSquared(point, intr.position)
                let angle = absDot(Vector(normal: intr.normal), -wi)
                pdf *= squaredDistance / angle
                return (intr, pdf)
        }
}

extension Sampleable where Self: Intersectable {

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from: I
        )
                throws -> FloatX
        {
                let ray = from.spawnRay(inDirection: direction)
                var tHit: FloatX = 0.0
                var interaction = SurfaceInteraction()
                try intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
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
