protocol SpectrumTexture: Texture {
        func evaluateSpectrum(at interaction: Interaction) -> Spectrum
}

extension SpectrumTexture {
        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                return evaluateSpectrum(at: interaction)
        }
}
