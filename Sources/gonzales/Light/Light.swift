///        A type that provides photons to the scene and can be sampled.
protocol Light {
        func sample(for reference: Interaction, u: Point2F) -> (
                radiance: Spectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        )
        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        func radianceFromInfinity(for ray: Ray) -> Spectrum
        var isDelta: Bool { get }
}
