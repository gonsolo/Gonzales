protocol FloatTexture: Texture {
        func evaluateFloat(at interaction: Interaction) -> FloatX
}

extension FloatTexture {
        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                return evaluateFloat(at: interaction)
        }
}
