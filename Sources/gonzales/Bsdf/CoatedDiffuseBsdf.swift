struct CoatedDiffuseBxdf: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                // TODO
                return reflectance / FloatX.pi
        }

        func albedo() -> RGBSpectrum { return reflectance }

        let reflectance: RGBSpectrum
        let roughness: (FloatX, FloatX)

        let topBxdf = Dielectric()
        let bottomBxdf = DiffuseBxdf()
}
