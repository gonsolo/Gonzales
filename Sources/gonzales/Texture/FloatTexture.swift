enum FloatTexture {
        case constantTexture(ConstantTexture<FloatX>)
        case openImageIoTexture(OpenImageIOTexture)

        // This should be not here: It forces FloatTexture to be indirect which forces it to be
        // stored on the heap.
        //case floatMixTexture(FloatMixTexture)

        func evaluateFloat(at interaction: any Interaction) -> FloatX {
                switch self {
                case .constantTexture(let constantTexture):
                        return constantTexture.evaluateFloat(at: interaction)
                //case .floatMixTexture(let floatMixTexture):
                //        return floatMixTexture.evaluateFloat(at: interaction)
                case .openImageIoTexture(let openImageIoTexture):
                        return openImageIoTexture.evaluateFloat(at: interaction)
                }
        }

        func evaluate(at interaction: any Interaction) -> any TextureEvaluation {
                return evaluateFloat(at: interaction)
        }
}
