import Foundation

struct ScaledTexture {

        func evaluateRgbSpectrum(at interaction: any Interaction) -> RgbSpectrum {
                guard let scale = scale.evaluate(at: interaction) as? FloatX else {
                        return black
                }
                guard let spectrum = texture.evaluate(at: interaction) as? RgbSpectrum else {
                        return black
                }
                return scale * spectrum
        }

        let scale: Texture
        let texture: Texture
}
