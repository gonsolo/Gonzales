import Foundation

struct Checkerboard {

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                guard let textureEven = textures.0.evaluate(at: interaction) as? RgbSpectrum else {
                        return black
                }
                guard let textureOdd = textures.1.evaluate(at: interaction) as? RgbSpectrum else {
                        return black
                }
                let uIndex = Int(floor(interaction.uvCoordinates[0] * scale.0))
                let vIndex = Int(floor(interaction.uvCoordinates[1] * scale.1))
                if (uIndex + vIndex) % 2 == 0 {
                        return textureEven
                } else {
                        return textureOdd
                }
        }

        let textures: (Texture, Texture)
        let scale: (FloatX, FloatX)
}
