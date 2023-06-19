import Foundation

final class MixTexture<T: TextureEvaluation>: Texture {

        init(textures: (Texture, Texture), amount: FloatX) {
                self.textures = textures
                self.amount = amount
        }

        func evaluate(at: Interaction) -> TextureEvaluation {
                // TODO
                return FloatX(1.0)
        }

        let textures: (Texture, Texture)
        let amount: FloatX
}

extension MixTexture: RGBSpectrumTexture where T == RGBSpectrum {
        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                return evaluate(at: interaction) as! RGBSpectrum
        }
}

extension MixTexture: FloatTexture where T == FloatX {
        func evaluateFloat(at interaction: Interaction) -> FloatX {
                return evaluate(at: interaction) as! FloatX
        }
}
