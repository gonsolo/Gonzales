import Foundation

final class ScaledTexture: RGBSpectrumTexture {

        init(scale: Texture, texture: Texture) {
                self.scale = scale
                self.texture = texture
        }

        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                guard let scale = scale.evaluate(at: interaction) as? FloatX else {
                        return black
                }
                guard let spectrum = texture.evaluate(at: interaction) as? RGBSpectrum else {
                        return black
                }
                return scale * spectrum
        }

        let scale: Texture
        let texture: Texture
}
