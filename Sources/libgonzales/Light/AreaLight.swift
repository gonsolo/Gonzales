import Foundation

final class AreaLight: Boundable, Intersectable, LightSource {

        init(brightness: RgbSpectrum, shape: ShapeType, alpha: Real, reverseOrientation: Bool, idx: Int) {
                self.brightness = brightness
                self.shape = shape
                self.alpha = alpha
                self.reverseOrientation = reverseOrientation
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
                -> Real {
                return shape.probabilityDensityFor(
                        scene: scene, samplingDirection: direction, from: reference)
        }

        func radianceFromInfinity(for _: Ray, arena _: TextureArena) -> RgbSpectrum { return black }

        func power(scene: Scene) -> Real {
                return brightness.average() * shape.area(scene: scene) * Real.pi
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return shape.worldBound(scene: scene)
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return shape.objectBound(scene: scene)
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) -> Bool {
                if alpha == 0 { return false }
                return shape.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                let interaction = shape.computeSurfaceInteraction(
                        scene: scene,
                        data: data,
                        worldRay: worldRay)
                if var unwrapped = interaction {
                        unwrapped.areaLightIndex = self.idx
                        if reverseOrientation {
                                unwrapped.normal = -unwrapped.normal
                                unwrapped.shadingNormal = -unwrapped.shadingNormal
                        }
                        return unwrapped
                }
                return nil
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if alpha == 0 { return nil }
                let interaction = shape.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
                if var unwrapped = interaction {
                        unwrapped.areaLightIndex = self.idx
                        if reverseOrientation {
                                unwrapped.normal = -unwrapped.normal
                                unwrapped.shadingNormal = -unwrapped.shadingNormal
                        }
                        return unwrapped
                }
                return nil
        }

        func getBsdf(interaction: SurfaceInteraction, arena _: TextureArena) -> DiffuseBsdf {
                return DiffuseBsdf(reflectance: white, bsdfFrame: BsdfFrame(interaction: interaction))
        }

        let shape: ShapeType
        let brightness: RgbSpectrum
        let alpha: Real
        let reverseOrientation: Bool
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
