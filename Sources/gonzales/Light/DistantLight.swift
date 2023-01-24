struct DistantLight: Light {

        init(lightToWorld: Transform, brightness: RGBSpectrum, direction: Vector) {
                self.brightness = brightness
                self.direction = normalized(lightToWorld * direction)
        }

        func sample(for reference: Interaction, u: Point2F) -> (
                radiance: RGBSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let outside = reference.position + direction * 2 * worldRadius
                let visibility = Visibility(
                        from: reference, to: SurfaceInteraction(position: outside))
                return (brightness, direction, pdf: 1, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                return 0
        }

        func radianceFromInfinity(for ray: Ray) -> RGBSpectrum { return black }

        var isDelta: Bool { return true }

        let direction: Vector
        let brightness: RGBSpectrum
        let worldRadius: FloatX = 100.0  // TODO
}

func createDistantLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> DistantLight
{
        let from = try parameters.findOnePoint(name: "from", else: origin)
        let to = try parameters.findOnePoint(name: "to", else: origin)
        guard let brightness = try parameters.findSpectrum(name: "L") as? RGBSpectrum else {
                throw ParameterError.missing(parameter: "L", function: #function)
        }
        let direction: Vector = from - to
        return DistantLight(
                lightToWorld: lightToWorld, brightness: brightness, direction: direction)
}
