struct ConstantTexture<T: TextureEvaluation> {

        func evaluate(at: any Interaction) -> any TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture where T == RgbSpectrum {

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                return value
        }
}

extension ConstantTexture where T == FloatX {
        func evaluateFloat(at interaction: any Interaction) -> FloatX {
                return value
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
