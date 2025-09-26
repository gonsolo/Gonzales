struct ConstantTexture<T: TextureEvaluation>: Sendable {

        func evaluate(at: InteractionType) -> any TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture where T == RgbSpectrum {

        func evaluateRgbSpectrum(at interaction: InteractionType) -> RgbSpectrum {
                return evaluate(at: interaction) as! RgbSpectrum
        }
}

extension ConstantTexture where T == FloatX {
        func evaluateFloat(at interaction: InteractionType) -> FloatX {
                return evaluate(at: interaction) as! FloatX
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
