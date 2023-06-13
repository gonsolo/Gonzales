struct DiffuseBsdf: GlobalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> RGBSpectrum { return reflectance }

        var reflectance: RGBSpectrum = white

        let bsdfFrame: BsdfFrame
}
