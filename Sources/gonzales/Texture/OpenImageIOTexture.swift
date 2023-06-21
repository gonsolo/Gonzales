import Foundation

final class OpenImageIOTexture: FloatTexture, RgbSpectrumTexture {

        enum TextureType {
                case float
                case rgb
        }

        init(path: String, type: String) {
                filename = path
                switch type {
                case "spectrum", "color":
                        textureType = .rgb
                case "float":
                        textureType = .float
                default:
                        fatalError("Unknown texture type in OpenImageIOTexture!")
                }
        }

        private func getTextureCoordinates(at interaction: Interaction) -> (s: FloatX, t: FloatX) {
                var (_, s) = modf(interaction.uv.x)
                var (_, t) = modf(interaction.uv.y)
                if s < 0 {
                        s = 1 + s
                }
                if t < 0 {
                        t = 1 + t
                }
                return (s, t)
        }

        func evaluateFloat(at interaction: Interaction) -> FloatX {
                let (s, t) = getTextureCoordinates(at: interaction)
                return OpenImageIOTextureSystem.shared.evaluate(filename: filename, s: s, t: t)
        }

        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                let (s, t) = getTextureCoordinates(at: interaction)
                return OpenImageIOTextureSystem.shared.evaluate(filename: filename, s: s, t: t)
        }

        func evaluate(at interaction: Interaction) -> TextureEvaluation {
                switch textureType {
                case .float:
                        return evaluateFloat(at: interaction)
                case .rgb:
                        return evaluateRgbSpectrum(at: interaction)
                }
        }

        let filename: String
        let textureType: TextureType
}
