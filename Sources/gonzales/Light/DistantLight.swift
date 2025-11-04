import Foundation

struct DistantLight {

        init(lightToWorld: Transform, brightness: RgbSpectrum, direction: Vector) {
                self.brightness = brightness
                self.direction = normalized(lightToWorld * direction)
        }

        func sample(point: Point, u: TwoRandomVariables, accelerator: Accelerator) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let outside = point + direction * 2 * worldRadius
                let visibility = Visibility(
                        from: point, to: outside)
                return (brightness, direction, pdf: 1, visibility)
        }

        func probabilityDensityFor(
                scene: Scene, samplingDirection direction: Vector, from reference: any Interaction
        )
                throws -> FloatX
        {
                return 0
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                let value = square(worldRadius) * FloatX.pi * brightness.average()
                return value
        }

        let direction: Vector
        let brightness: RgbSpectrum
        let worldRadius: FloatX = 100.0  // TODO
}

func createDistantLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> DistantLight
{
        let from = try parameters.findOnePoint(name: "from", else: origin)
        let to = try parameters.findOnePoint(name: "to", else: origin)
        guard let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum else {
                throw ParameterError.missing(parameter: "L", function: #function)
        }
        let direction: Vector = from - to
        return DistantLight(
                lightToWorld: lightToWorld, brightness: brightness, direction: direction)
}
