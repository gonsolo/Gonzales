struct DiffuseBsdf: GlobalBsdf {

        func evaluateLocal(wo _: Vector, wi _: Vector) -> RgbSpectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> RgbSpectrum { return reflectance }

        var reflectance: RgbSpectrum = white

        let bsdfFrame: BsdfFrame
}
