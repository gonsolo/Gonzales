protocol RgbSpectrumTexture: Texture {
        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum
}

extension RgbSpectrumTexture {
        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                return evaluateRgbSpectrum(at: interaction)
        }
}
