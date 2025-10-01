indirect enum FloatTexture {
        case constantTexture(ConstantTexture<FloatX>)
        case floatMixTexture(FloatMixTexture)
        case openImageIoTexture(OpenImageIOTexture)

        func evaluateFloat(at interaction: any Interaction) -> FloatX {
                switch self {
                case .constantTexture(let constantTexture):
                        return constantTexture.evaluateFloat(at: interaction)
                case .floatMixTexture(let floatMixTexture):
                        return floatMixTexture.evaluateFloat(at: interaction)
                case .openImageIoTexture(let openImageIoTexture):
                        return openImageIoTexture.evaluateFloat(at: interaction)
                }
        }

        func evaluate(at interaction: any Interaction) -> any TextureEvaluation {
                return evaluateFloat(at: interaction)
        }
}
