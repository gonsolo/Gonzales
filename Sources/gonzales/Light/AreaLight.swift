import Foundation

struct AreaLight: Boundable, Intersectable, Sendable {

        init(brightness: RgbSpectrum, shape: ShapeType, alpha: FloatX, idx: Int) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
                self.idx = idx
        }

        func emittedRadiance(from interaction: any Interaction, inDirection direction: Vector)
                -> RgbSpectrum
        {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(point: Point, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let (shapeInteraction, pdf) = shape.sample(point: point, u: u)
                let direction: Vector = normalized(shapeInteraction.position - point)
                assert(!direction.isNaN)
                let visibility = Visibility(from: point, to: shapeInteraction.position)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return (radiance, direction, pdf, visibility)
        }

        func probabilityDensityFor<I: Interaction>(
                samplingDirection direction: Vector,
                from reference: I
        )
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

        func getIntersectionData(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> TriangleIntersection? {
                if alpha == 0 { return nil }
                return try shape.getIntersectionData(ray: worldRay, tHit: &tHit)
        }

        func computeSurfaceInteraction(
                data: TriangleIntersection?,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                if alpha == 0 { return }
                shape.computeSurfaceInteraction(
                        data: data,
                        worldRay: worldRay,
                        interaction: &interaction)
                interaction.areaLight = self
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.intersect(
                        ray: ray,
                        tHit: &tHit)
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                if alpha == 0 { return }
                try shape.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                interaction.areaLight = self
        }

        func getBsdf(interaction: SurfaceInteraction) -> DiffuseBsdf {
                let diffuse = Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white))))
                return diffuse.getBsdf(interaction: interaction)
        }

        let shape: ShapeType
        let brightness: RgbSpectrum
        let alpha: FloatX
        let idx: Int
}

extension AreaLight: Equatable {
        static func == (lhs: AreaLight, rhs: AreaLight) -> Bool {
                return
                        // TODO lhs.shape == rhs.shape &&
                        lhs.brightness == rhs.brightness && lhs.alpha == rhs.alpha
        }
}
