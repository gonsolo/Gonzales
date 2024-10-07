struct ConstantTexture<T: TextureEvaluation>: Sendable {

        func evaluate(at: Interaction) -> TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture where T == RgbSpectrum {

        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                return evaluate(at: interaction) as! RgbSpectrum
        }
}

extension ConstantTexture where T == FloatX {
        func evaluateFloat(at interaction: Interaction) -> FloatX {
                return evaluate(at: interaction) as! FloatX
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
