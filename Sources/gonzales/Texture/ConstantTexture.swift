final class ConstantTexture<T: TextureEvaluation>: Texture {

        init(value: T) {
                self.value = value
        }

        func evaluate(at: Interaction) -> TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture: RgbSpectrumTexture where T == RgbSpectrum {
        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                return evaluate(at: interaction) as! RgbSpectrum
        }
}

extension ConstantTexture: FloatTexture where T == FloatX {
        func evaluateFloat(at interaction: Interaction) -> FloatX {
                return evaluate(at: interaction) as! FloatX
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
