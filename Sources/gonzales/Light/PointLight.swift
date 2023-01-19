struct PointLight: Light {

        init(lightToWorld: Transform, intensity: RGBSpectrum) {
                position = lightToWorld * origin
                self.intensity = intensity
        }

        func sample(for reference: Interaction, u: Point2F) -> (
                radiance: RGBSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let direction: Vector = normalized(position - reference.position)
                let pdf: FloatX = 1.0
                let visibility = Visibility(
                        from: reference, to: SurfaceInteraction(position: position))
                let distance2 = distanceSquared(position, reference.position)
                let radiance = intensity / distance2
                return (radiance, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                return 0
        }

        func radianceFromInfinity(for ray: Ray) -> RGBSpectrum { return black }

        var isDelta: Bool { return true }

        let position: Point
        let intensity: RGBSpectrum
}

func createPointLight(lightToWorld: Transform, parameters: ParameterDictionary) throws -> PointLight
{
        guard let intensity = try parameters.findSpectrum(name: "I") as? RGBSpectrum else {
                throw ParameterError.missing(parameter: "I")
        }
        guard let scale = try parameters.findSpectrum(name: "scale", else: white) as? RGBSpectrum
        else {
                throw ParameterError.missing(parameter: "scale")
        }
        return PointLight(lightToWorld: lightToWorld, intensity: scale * intensity)
}
