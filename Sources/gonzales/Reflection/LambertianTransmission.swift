struct LambertianTransmission: BxDF {

        init(reflectance: Spectrum) {
                self.reflectance = reflectance
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                return reflectance / FloatX.pi
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                if !sameHemisphere(wo, wi) {
                        return absCosTheta(wi) / FloatX.pi
                } else {
                        return 0
                }
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
}
