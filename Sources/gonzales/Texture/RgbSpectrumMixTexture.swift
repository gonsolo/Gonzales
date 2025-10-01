struct RgbSpectrumMixTexture {

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                let value0 = textures.0.evaluateRgbSpectrum(at: interaction)
                let value1 = textures.1.evaluateRgbSpectrum(at: interaction)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (RgbSpectrumTexture, RgbSpectrumTexture)
        let amount: FloatX
}
