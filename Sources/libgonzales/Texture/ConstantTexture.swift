struct ConstantTexture<T: TextureEvaluation> {

        func evaluate(at _: any Interaction) -> any TextureEvaluation {
                return value
        }

        var value: T
}

extension ConstantTexture where T == RgbSpectrum {

        func evaluateRgbSpectrum(at _: any Interaction) -> RgbSpectrum {
                return value
        }
}

extension ConstantTexture where T == Real {
        func evaluateFloat(at _: any Interaction) -> Real {
                return value
        }
}

extension ConstantTexture: CustomStringConvertible {
        var description: String {
                return "\(value)"
        }
}
