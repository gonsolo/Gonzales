struct ScaledTextureRgb: Sendable {
        let tex: RgbSpectrumTexture
        let scale: FloatTexture

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                return scale.evaluateFloat(at: interaction) * tex.evaluateRgbSpectrum(at: interaction)
        }
}

struct ScaledTextureFloat: Sendable {
        let tex: FloatTexture
        let scale: FloatTexture

        func evaluateFloat(at interaction: any Interaction) -> Real {
                return scale.evaluateFloat(at: interaction) * tex.evaluateFloat(at: interaction)
        }
}
