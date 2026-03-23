struct FloatMixTexture: Sendable {

        func evaluateFloat(at interaction: any Interaction, arena: TextureArena) -> Float {
                let value0 = arena.floatTextures[textures.0].evaluateFloat(at: interaction, arena: arena)
                let value1 = arena.floatTextures[textures.1].evaluateFloat(at: interaction, arena: arena)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (Int, Int)
        let amount: Real
}
