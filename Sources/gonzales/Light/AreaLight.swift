import Foundation

struct AreaLight: Boundable, Intersectable {

        init(brightness: RgbSpectrum, shape: Shape, alpha: FloatX) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
        }

        func emittedRadiance(from interaction: Interaction, inDirection direction: Vector)
                -> RgbSpectrum
        {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(for ref: Interaction, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let (shapeInteraction, pdf) = shape.sample(ref: ref, u: u)
                let direction: Vector = normalized(shapeInteraction.position - ref.position)
                assert(!direction.isNaN)
                let visibility = Visibility(from: ref, to: shapeInteraction)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return (radiance, direction, pdf, visibility)
        }

        @MainActor
        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                async throws -> FloatX
        {
                return try await shape.probabilityDensityFor(
                        samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                return brightness.average() * shape.area() * FloatX.pi
        }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                if alpha == 0 { return }
                try await shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                interaction.areaLight = self
        }

        @MainActor
        func getBsdf(interaction: Interaction) -> GlobalBsdf {
                let diffuse = Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white))))
                return diffuse.getBsdf(interaction: interaction)
        }

        let shape: Shape
        let brightness: RgbSpectrum
        let alpha: FloatX
}
