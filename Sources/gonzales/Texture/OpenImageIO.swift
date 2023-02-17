import Foundation
import openImageIOBridge

final class OpenImageIOTextureSystem {

        static let shared = OpenImageIOTextureSystem()

        func evaluate(filename: String, s: Float, t: Float) -> RGBSpectrum {
                let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 3)
                let successful = texture(filename, s, t, pointer)
                if !successful {
                        warnOnce("Could not evaluate texture!")
                        return black
                }
                let result = RGBSpectrum(r: pointer[0], g: pointer[1], b: pointer[2])
                return result
        }

        private init() {
                createTextureSystem()
        }

        deinit {
                destroyTextureSystem()
        }
}

final class OpenImageIOTexture: RGBSpectrumTexture {

        init(path: String) {
                filename = path
        }

        let filename: String

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

}
