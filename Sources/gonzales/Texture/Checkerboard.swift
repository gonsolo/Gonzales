import Foundation

final class Checkerboard: RGBSpectrumTexture {

        init(textures: (Texture, Texture), scale: (FloatX, FloatX)) {
                self.textures = textures
                self.scale = scale
        }

        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                guard let textureEven = textures.0.evaluate(at: interaction) as? RGBSpectrum else {
                        return black
                }
                guard let textureOdd = textures.1.evaluate(at: interaction) as? RGBSpectrum else {
                        return black
                }
                return RGBSpectrum(r: interaction.uv[0], g: interaction.uv[1], b: 0)
                //let u = Int(floor(interaction.uv[0] * scale.0))
                //let v = Int(floor(interaction.uv[1] * scale.1))
                //if (u + v) % 2 == 0 {
                //        return textureEven
                //} else {
                //        return textureOdd
                //}
        }

        let textures: (Texture, Texture)
        let scale: (FloatX, FloatX)
}
