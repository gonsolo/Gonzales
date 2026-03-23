enum RgbSpectrumTexture: Sendable {

        // case checkerboard(Checkerboard)
        case constantTexture(ConstantTexture<RgbSpectrum>)
        case openImageIoTexture(OpenImageIOTexture)
        case ptex(Ptex)
        case scaledTexture(ScaledTextureRgb)
        // case rgbSpectrumMixTexture(RgbSpectrumMixTexture)

        func evaluateRgbSpectrum(at interaction: any Interaction, arena: TextureArena) -> RgbSpectrum {
                switch self {
                case .constantTexture(let value):
                        return value.evaluateRgbSpectrum(at: interaction, arena: arena)
                case .openImageIoTexture(let value):
                        return value.evaluateRgbSpectrum(at: interaction, arena: arena)
                case .ptex(let value): return value.evaluateRgbSpectrum(at: interaction, arena: arena)
                case .scaledTexture(let value):
                        return value.evaluateRgbSpectrum(at: interaction, arena: arena)
                }
        }

        func evaluate(at interaction: any Interaction, arena: TextureArena) -> any TextureEvaluation {
                return evaluateRgbSpectrum(at: interaction, arena: arena)
        }
}
