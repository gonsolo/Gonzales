final class LambertianReflection: BxDF {

        init(reflectance: Spectrum) {
                self.reflectance = reflectance
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
}
