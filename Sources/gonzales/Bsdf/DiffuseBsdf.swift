struct DiffuseBsdf: LocalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func albedoLocal() -> RGBSpectrum { return reflectance }

        var reflectance: RGBSpectrum = white
}
