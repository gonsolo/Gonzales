protocol RGBSpectrumTexture: Texture {
        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum
}

extension RGBSpectrumTexture {
        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                return evaluateRGBSpectrum(at: interaction)
        }
}
