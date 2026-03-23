import Foundation
import ptexBridge

struct PtexCache {

        init(ptexMemory: Int) {
                initPtexCache(ptexMemory)
        }
}

struct Ptex {

        init(path: String) {
                self.path = path
                initPtexTexture(path)
        }

        func evaluateRgbSpectrum(at interaction: any Interaction, arena: TextureArena) -> RgbSpectrum {
                let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 3)
                evaluatePtex(
                        path, interaction.faceIndex, Float(interaction.uvCoordinates[0]),
                        Float(interaction.uvCoordinates[1]), pointer)
                let spectrum = RgbSpectrum(
                        red: Real(pointer[0]), green: Real(pointer[1]), blue: Real(pointer[2]))
                return gammaSrgbToLinear(light: spectrum)
        }

        let path: String
}
