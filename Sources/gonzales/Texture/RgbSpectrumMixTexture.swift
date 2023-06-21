final class RgbSpectrumMixTexture: RgbSpectrumTexture {

        init(textures: (RgbSpectrumTexture, RgbSpectrumTexture), amount: FloatX) {
                self.textures = textures
                self.amount = amount
        }

        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                let value0 = textures.0.evaluateRgbSpectrum(at: interaction)
                let value1 = textures.1.evaluateRgbSpectrum(at: interaction)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (RgbSpectrumTexture, RgbSpectrumTexture)
        let amount: FloatX
}
