enum Texture: Sendable {
        case floatTexture(FloatTexture)
        case rgbSpectrumTexture(RgbSpectrumTexture)
        case dummy

        func evaluate(at interaction: any Interaction) -> any TextureEvaluation {
                switch self {
                case .floatTexture(let floatTexture):
                        return floatTexture.evaluate(at: interaction)
                case .rgbSpectrumTexture(let rgbSpectrumTexture):
                        return rgbSpectrumTexture.evaluate(at: interaction)
                case .dummy:
                        return black
                }
        }
}
