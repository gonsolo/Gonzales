import Foundation

struct PointLight {

        init(lightToWorld: Transform, intensity: RgbSpectrum) {
                position = lightToWorld * origin
                self.intensity = intensity
        }

        func sample(point: Point, u: TwoRandomVariables, accelerator: Accelerator) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let direction: Vector = normalized(position - point)
                let pdf: FloatX = 1.0
                let visibility = Visibility(from: point, to: position)
                let distance2 = distanceSquared(position, point)
                let radiance = intensity / distance2
                return (radiance, direction, pdf, visibility)
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
