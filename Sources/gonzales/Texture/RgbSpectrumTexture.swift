indirect enum RgbSpectrumTexture: Sendable {

        case checkerboard(Checkerboard)
        case constantTexture(ConstantTexture<RgbSpectrum>)
        case openImageIoTexture(OpenImageIOTexture)
        case ptex(Ptex)
        case rgbSpectrumMixTexture(RgbSpectrumMixTexture)
        case scaledTexture(ScaledTexture)

        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                switch self {
                case .checkerboard(let checkerboard):
                        return checkerboard.evaluateRgbSpectrum(at: interaction)
                case .constantTexture(let constantTexture):
                        return constantTexture.evaluateRgbSpectrum(at: interaction)
                case .openImageIoTexture(let openImageIoTexture):
                        return openImageIoTexture.evaluateRgbSpectrum(at: interaction)
                case .ptex(let ptex):
                        return ptex.evaluateRgbSpectrum(at: interaction)
                case .rgbSpectrumMixTexture(let rgbSpectrumMixTexture):
                        return rgbSpectrumMixTexture.evaluateRgbSpectrum(at: interaction)
                case .scaledTexture(let scaledTexture):
                        return scaledTexture.evaluateRgbSpectrum(at: interaction)
                }
        }

        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                return evaluateRgbSpectrum(at: interaction)
        }
}
