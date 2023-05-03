import Foundation

final class ScaledTexture: RGBSpectrumTexture {

        init(scale: FloatX, texture: Texture) {
                self.scale = scale
                self.texture = texture
        }

        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                guard let spectrum = texture.evaluate(at: interaction) as? RGBSpectrum else {
                        return black
                }
                return scale * spectrum
        }

        let scale: FloatX
        let texture: Texture
}
