import Foundation

struct AreaLight: Light, Boundable, Intersectable, Material {

        init(brightness: RGBSpectrum, shape: Shape, alpha: FloatX) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
        }

        func emittedRadiance(from interaction: Interaction, inDirection direction: Vector)
                -> RGBSpectrum
        {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(for ref: Interaction, u: TwoRandomVariables) -> (
                radiance: RGBSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let (shapeInteraction, pdf) = shape.sample(ref: ref, u: u)
                let direction: Vector = normalized(shapeInteraction.position - ref.position)
                assert(!direction.isNaN)
                let visibility = Visibility(from: ref, to: shapeInteraction)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return (radiance, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                return try shape.probabilityDensityFor(
                        samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for ray: Ray) -> RGBSpectrum { return black }

        func power() -> Measurement<UnitPower> {
                return Measurement(
                        value: Double(brightness.average() * shape.area() * FloatX.pi),
                        unit: UnitPower.watts)
        }

        var isDelta: Bool { return false }

        func worldBound() -> Bounds3f {
                return shape.worldBound()
        }

        func objectBound() -> Bounds3f {
                return shape.objectBound()
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                if alpha == 0 { return }
                try shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        material: material,
                        interaction: &interaction)
                interaction.areaLight = self
        }

        func getGlobalBsdf(interaction: SurfaceInteraction) -> GlobalBsdf {
                let diffuse = Diffuse(reflectance: ConstantTexture(value: white))
                return diffuse.getGlobalBsdf(interaction: interaction)
        }

        let shape: Shape
        let brightness: RGBSpectrum
        let alpha: FloatX
}
