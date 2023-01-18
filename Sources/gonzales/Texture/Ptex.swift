import Foundation
import ptex

struct PtexCache {
        init() {
                initPtexCache(ptexMemory)
        }
}

final class Ptex: RGBSpectrumTexture {

        init(path: String) {
                self.path = path
                initPtexTexture(path)
        }

        func evaluateRGBSpectrum(at interaction: Interaction) -> RGBSpectrum {
                let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 3)
                evaluatePtex(
                        path, interaction.faceIndex, Float(interaction.uv[0]),
                        Float(interaction.uv[1]), pointer)
                let spectrum = RGBSpectrum(
                        r: FloatX(pointer[0]), g: FloatX(pointer[1]), b: FloatX(pointer[2]))
                return gammaSrgbToLinear(light: spectrum)
        }

        var path: String
}
