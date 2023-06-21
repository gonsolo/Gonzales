final class FloatMixTexture: FloatTexture {

        init(textures: (FloatTexture, FloatTexture), amount: FloatX) {
                self.textures = textures
                self.amount = amount
        }

        func evaluateFloat(at interaction: Interaction) -> Float {
                let value0 = textures.0.evaluateFloat(at: interaction)
                let value1 = textures.1.evaluateFloat(at: interaction)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (FloatTexture, FloatTexture)
        let amount: FloatX
}
