struct LambertianReflection: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> Spectrum { return reflectance }

        var reflectance: Spectrum
}
