struct RgbSpectrumMixTexture: Sendable {

        func evaluateRgbSpectrum(at interaction: any Interaction, arena: TextureArena) -> RgbSpectrum {
                let value0 = arena.rgbTextures[textures.0].evaluateRgbSpectrum(at: interaction, arena: arena)
                let value1 = arena.rgbTextures[textures.1].evaluateRgbSpectrum(at: interaction, arena: arena)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (Int, Int)
        let amount: Real
}
