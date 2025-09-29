import Foundation

struct AreaLight: Boundable, Intersectable, Sendable {

        init(brightness: RgbSpectrum, shape: any Shape, alpha: FloatX) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
        }

        func emittedRadiance(from interaction: InteractionType, inDirection direction: Vector)
                -> RgbSpectrum
        {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(for ref: InteractionType, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let (shapeInteraction, pdf) = shape.sample(ref: ref, u: u)
                let direction: Vector = normalized(shapeInteraction.position - ref.position)
                assert(!direction.isNaN)
                let visibility = Visibility(from: ref, to: shapeInteraction)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return (radiance, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: InteractionType)
                throws -> FloatX
        {
                return try shape.probabilityDensityFor(
                        samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                return brightness.average() * shape.area() * FloatX.pi
        }

        func worldBound() async -> Bounds3f {
                return await shape.worldBound()
        }

        func objectBound() async -> Bounds3f {
                return await shape.objectBound()
        }

        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitive? {
                if alpha == 0 { return nil }
                return try shape.intersect_lean(
                        ray: ray,
                        tHit: &tHit)
        }

        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction
        //) throws {
        //        if alpha == 0 { return }
        //        try shape.intersect(
        //                ray: ray,
        //                tHit: &tHit,
        //                interaction: &interaction)
        //        interaction.areaLight = self
        //}

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction {
                var interaction = SurfaceInteraction()
                if alpha == 0 { return interaction }
                interaction = try shape.computeInteraction(
                        ray: ray,
                        tHit: &tHit)
                interaction.areaLight = self
                return interaction
        }
        func getBsdf(interaction: InteractionType) -> DiffuseBsdf {
                let diffuse = Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white))))
                return diffuse.getBsdf(interaction: interaction)
        }

        let shape: any Shape
        let brightness: RgbSpectrum
        let alpha: FloatX
}
