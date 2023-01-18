struct LambertianReflection: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> RGBSpectrum { return reflectance }

        var reflectance: RGBSpectrum
}
