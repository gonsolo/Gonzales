struct ScaledTextureRgb: Sendable {
        let tex: Int
        let scale: Int

        func evaluateRgbSpectrum(at interaction: any Interaction, arena: TextureArena) -> RgbSpectrum {
                return arena.floatTextures[scale].evaluateFloat(at: interaction, arena: arena) * arena.rgbTextures[tex].evaluateRgbSpectrum(at: interaction, arena: arena)
        }
}

struct ScaledTextureFloat: Sendable {
        let tex: Int
        let scale: Int

        func evaluateFloat(at interaction: any Interaction, arena: TextureArena) -> Real {
                return arena.floatTextures[scale].evaluateFloat(at: interaction, arena: arena) * arena.floatTextures[tex].evaluateFloat(at: interaction, arena: arena)
        }
}
