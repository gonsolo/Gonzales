import Foundation

struct DistantLight {

        init(lightToWorld: Transform, brightness: RgbSpectrum, direction: Vector) {
                self.brightness = brightness
                self.direction = normalized(lightToWorld * direction)
        }

        func sample(point: Point, samples _: TwoRandomVariables, accelerator _: Accelerator)
                -> LightSample {  // radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
                let outside = point + direction * 2 * worldRadius
                let visibility = Visibility(
                        from: point, target: outside)
                return LightSample(radiance: brightness, direction: direction, pdf: 1, visibility: visibility)
        }

        func probabilityDensityFor(
                scene _: Scene, samplingDirection _: Vector, from _: any Interaction
        )
                throws -> FloatX {
                return 0
        }

        func radianceFromInfinity(for _: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                let value = square(worldRadius) * FloatX.pi * brightness.average()
                return value
        }

        let direction: Vector
        let brightness: RgbSpectrum
        let worldRadius: FloatX = 100.0
}

func createDistantLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> DistantLight {
        let from = try parameters.findOnePoint(name: "from", else: origin)
        let target = try parameters.findOnePoint(name: "to", else: origin)
        guard let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum else {
                throw ParameterError.missing(parameter: "L", function: #function)
        }
        let direction: Vector = from - target
        return DistantLight(
                lightToWorld: lightToWorld, brightness: brightness, direction: direction)
}
