import Foundation

final class FloatMixTexture: FloatTexture {

        init(textures: (FloatTexture, FloatTexture), amount: FloatX) {
                self.textures = textures
                self.amount = amount
        }

        func evaluateFloat(at: Interaction) -> Float {
                // TODO
                return FloatX(1.0)
        }

        let textures: (FloatTexture, FloatTexture)
        let amount: FloatX
}

final class RgbSpectrumMixTexture: RgbSpectrumTexture {

        init(textures: (RgbSpectrumTexture, RgbSpectrumTexture), amount: FloatX) {
                self.textures = textures
                self.amount = amount
        }

        func evaluateRgbSpectrum(at: Interaction) -> RgbSpectrum {
                // TODO
                return white
        }

        let textures: (RgbSpectrumTexture, RgbSpectrumTexture)
        let amount: FloatX
}
