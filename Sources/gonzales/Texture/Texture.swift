enum Texture: Sendable {
        case floatTexture(FloatTexture)
        case rgbSpectrumTexture(RgbSpectrumTexture)

        func evaluate(at interaction: any Interaction) -> any TextureEvaluation {
                switch self {
                case .floatTexture(let floatTexture):
                        return floatTexture.evaluate(at: interaction)
                case .rgbSpectrumTexture(let rgbSpectrumTexture):
                        return rgbSpectrumTexture.evaluate(at: interaction)
                }
        }
}
