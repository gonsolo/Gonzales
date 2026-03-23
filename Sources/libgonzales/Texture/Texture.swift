enum Texture: Sendable {
        var index: Int {
                switch self {
                case .floatTexture(let i): return i
                case .rgbSpectrumTexture(let i): return i
                }
        }
        case floatTexture(Int)

    func evaluateFloat(at interaction: SurfaceInteraction, arena: TextureArena) -> Real {
        switch self {
        case .floatTexture(let index):
            return arena.floatTextures[index].evaluateFloat(at: interaction, arena: arena)
        case .rgbSpectrumTexture(let index):
            return arena.rgbTextures[index].evaluateRgbSpectrum(at: interaction, arena: arena).y
        }
    }

    func evaluateRgbSpectrum(at interaction: SurfaceInteraction, arena: TextureArena) -> RgbSpectrum {
        switch self {
        case .floatTexture(let index):
            let val = arena.floatTextures[index].evaluateFloat(at: interaction, arena: arena)
            return RgbSpectrum(intensity: val)
        case .rgbSpectrumTexture(let index):
            return arena.rgbTextures[index].evaluateRgbSpectrum(at: interaction, arena: arena)
        }
    }
        case rgbSpectrumTexture(Int)

    func evaluate(at interaction: any Interaction, arena: TextureArena) -> any TextureEvaluation {
        switch self {
        case .floatTexture(let index):
            return arena.floatTextures[index].evaluate(at: interaction, arena: arena)
        case .rgbSpectrumTexture(let index):
            return arena.rgbTextures[index].evaluate(at: interaction, arena: arena)
        }
    }
}
