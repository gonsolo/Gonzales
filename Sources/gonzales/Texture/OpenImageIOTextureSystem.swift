import openImageIOBridge

final class OpenImageIOTextureSystem {

        private init() {
                createTextureSystem()
        }

        deinit {
                destroyTextureSystem()
        }

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

        static let shared = OpenImageIOTextureSystem()
}
