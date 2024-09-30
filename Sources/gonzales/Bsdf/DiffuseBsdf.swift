struct DiffuseBsdf: GlobalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) async -> RgbSpectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> RgbSpectrum { return reflectance }

        var reflectance: RgbSpectrum = white

        let bsdfFrame: BsdfFrame
}
