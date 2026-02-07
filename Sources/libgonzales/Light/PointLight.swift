import Foundation

struct PointLight {

        init(lightToWorld: Transform, intensity: RgbSpectrum) {
                position = lightToWorld * origin
                self.intensity = intensity
        }

        func sample(point: Point, samples _: TwoRandomVariables, accelerator _: Accelerator) -> LightSample {
                let direction: Vector = normalized(position - point)
                let pdf: FloatX = 1.0
                let visibility = Visibility(from: point, target: position)
                let distance2 = distanceSquared(position, point)
                let radiance = intensity / distance2
                return LightSample(radiance: radiance, direction: direction, pdf: pdf, visibility: visibility)
        }

        func probabilityDensityFor(
                scene _: Scene, samplingDirection _: Vector, from _: any Interaction
        )
                throws -> FloatX {
                return 0
        }

        func radianceFromInfinity(for _: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                return intensity.average() * 4 * FloatX.pi
        }

        let position: Point
        let intensity: RgbSpectrum
}

func createPointLight(lightToWorld: Transform, parameters: ParameterDictionary) throws -> PointLight {
        guard let intensity = try parameters.findSpectrum(name: "I") as? RgbSpectrum else {
                throw ParameterError.missing(parameter: "I", function: #function)
        }
        guard let scale = try parameters.findSpectrum(name: "scale", else: white) as? RgbSpectrum
        else {
                throw ParameterError.missing(parameter: "scale", function: #function)
        }
        return PointLight(lightToWorld: lightToWorld, intensity: scale * intensity)
}
