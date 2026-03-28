import Foundation

struct PointLight: LightSource {

        init(lightToWorld: Transform, intensity: RgbSpectrum) {
                position = lightToWorld * origin
                self.intensity = intensity
        }

        func sample(
                point: Point, samples _: TwoRandomVariables,
                accelerator _: Accelerator, scene _: Scene
        ) -> LightSample {
                let direction: Vector = normalized(position - point)
                let pdf: Real = 1.0
                let visibility = Visibility(from: point, target: position)
                let distance2 = distanceSquared(position, point)
                let radiance = intensity / distance2
                return LightSample(radiance: radiance, direction: direction, pdf: pdf, visibility: visibility)
        }

        func probabilityDensityFor<I: Interaction>(
                scene _: Scene, samplingDirection _: Vector, from _: I
        )
                -> Real
        {
                return 0
        }

        func radianceFromInfinity(for _: Ray, arena _: TextureArena) -> RgbSpectrum { return black }

        func power(scene _: Scene) -> Real {
                return intensity.average() * 4 * Real.pi
        }

        let position: Point
        let intensity: RgbSpectrum
}

extension PointLight {
        static func create(lightToWorld: Transform, parameters: ParameterDictionary) throws -> PointLight {
                guard let intensity = try parameters.findSpectrum(name: "I") as? RgbSpectrum else {
                        throw ParameterError.missing(parameter: "I", function: #function)
                }
                guard let scale = try parameters.findSpectrum(name: "scale", else: white) as? RgbSpectrum
                else {
                        throw ParameterError.missing(parameter: "scale", function: #function)
                }
                return PointLight(lightToWorld: lightToWorld, intensity: scale * intensity)
        }
}

extension PointLight {
        var isDelta: Bool { true }
}
