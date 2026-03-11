import Foundation

struct AreaLight: Boundable, Intersectable, LightSource {

        init(brightness: RgbSpectrum, shape: ShapeType, alpha: FloatX, idx: Int) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
                self.idx = idx
        }

        func emittedRadiance(from interaction: any Interaction, inDirection direction: Vector)
                -> RgbSpectrum {
                return dot(Vector(normal: interaction.normal), direction) > 0 ? brightness : black
        }

        func sample(point: Point, samples: TwoRandomVariables, accelerator _: Accelerator, scene: Scene)
                -> LightSample {
                let (shapeInteraction, pdf) = shape.sample(point: point, samples: samples, scene: scene)
                let direction: Vector = normalized(shapeInteraction.position - point)
                assert(!direction.isNaN)
                let visibility = Visibility(from: point, target: shapeInteraction.position)
                let radiance = emittedRadiance(from: shapeInteraction, inDirection: -direction)
                return LightSample(radiance: radiance, direction: direction, pdf: pdf, visibility: visibility)
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from reference: I
        )
                throws -> FloatX {
                return try shape.probabilityDensityFor(
                        scene: scene, samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for _: Ray) -> RgbSpectrum { return black }

        func power(scene: Scene) -> FloatX {
                return brightness.average() * shape.area(scene: scene) * FloatX.pi
        }

        func worldBound(scene: Scene) async -> Bounds3f {
                return shape.worldBound(scene: scene)
        }

        func objectBound(scene: Scene) async -> Bounds3f {
                return shape.objectBound(scene: scene)
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection
        ) throws -> Bool {
                if alpha == 0 { return false }
                return try shape.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                var interaction = shape.computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay)
                interaction?.areaLight = self
                return interaction
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                var interaction = try shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
                interaction?.areaLight = self
                return interaction
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
                        lhs.brightness == rhs.brightness && lhs.alpha == rhs.alpha
        }
}

extension AreaLight {
        var isDelta: Bool { false }
}
