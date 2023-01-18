struct LambertianTransmission: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                if !sameHemisphere(wo, wi) {
                        return absCosTheta(wi) / FloatX.pi
                } else {
                        return 0
                }
        }

        func albedo() -> RGBSpectrum { return reflectance }

        var reflectance: RGBSpectrum
}
