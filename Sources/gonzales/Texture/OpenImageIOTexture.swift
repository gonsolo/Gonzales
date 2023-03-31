import Foundation

final class OpenImageIOTexture: RGBSpectrumTexture {

        init(path: String) {
                filename = path
        }

        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                var (_, s) = modf(interaction.uv.x)
                var (_, t) = modf(interaction.uv.y)
                if s < 0 {
                        s = 1 + s
                }
                if t < 0 {
                        t = 1 + t
                }
                return OpenImageIOTextureSystem.shared.evaluate(filename: filename, s: s, t: t)
        }

        let filename: String
}
