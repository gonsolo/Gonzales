import Foundation

enum TextureError: Error {
        case unknownType(String)
}

struct OpenImageIOTexture {

        enum TextureType {
                case float
                case rgb
        }

        init(path: String, type: String) throws {
                filename = path
                switch type {
                case "spectrum", "color":
                        textureType = .rgb
                case "float":
                        textureType = .float
                default:
                        throw TextureError.unknownType(type)
                }
        }

        private func getTextureCoordinates(at interaction: any Interaction) -> (s: Real, t: Real) {
                var (_, s) = modf(interaction.uvCoordinates.x)
                var (_, t) = modf(interaction.uvCoordinates.y)
                if s < 0 {
                        s = 1 + s
                }
                if t < 0 {
                        t = 1 + t
                }
                return (s, t)
        }

        func evaluateFloat(at interaction: any Interaction) -> Real {
                let (s, t) = getTextureCoordinates(at: interaction)
                return OpenImageIOTextureSystem.shared.evaluate(filename: filename, s: s, t: t)
        }

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                let (s, t) = getTextureCoordinates(at: interaction)
                let value: RgbSpectrum = OpenImageIOTextureSystem.shared.evaluate(
                        filename: filename, s: s, t: t)
                let lower = filename.lowercased()
                if lower.hasSuffix(".png") || lower.hasSuffix(".jpg") || lower.hasSuffix(".jpeg") || lower.hasSuffix(".tga") {
                        return gammaSrgbToLinear(light: value)
                }
                return value
        }

        func evaluate(at interaction: any Interaction) -> any TextureEvaluation {
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
