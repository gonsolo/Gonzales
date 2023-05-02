final class ConstantTexture<T: TextureEvaluation>: Texture {

        init(value: T) {
                self.value = value
        }

        func evaluate(at: Interaction) -> TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture: RGBSpectrumTexture where T == RGBSpectrum {
        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                return evaluate(at: interaction) as! RGBSpectrum
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
