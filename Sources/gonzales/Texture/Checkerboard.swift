import Foundation

struct Checkerboard {

        @MainActor
        func evaluateRgbSpectrum(at interaction: Interaction) -> RgbSpectrum {
                guard let textureEven = textures.0.evaluate(at: interaction) as? RgbSpectrum else {
                        return black
                }
                guard let textureOdd = textures.1.evaluate(at: interaction) as? RgbSpectrum else {
                        return black
                }
                let u = Int(floor(interaction.uv[0] * scale.0))
                let v = Int(floor(interaction.uv[1] * scale.1))
                if (u + v) % 2 == 0 {
                        return textureEven
                } else {
                        return textureOdd
                }
        }

        let textures: (Texture, Texture)
        let scale: (FloatX, FloatX)
}
