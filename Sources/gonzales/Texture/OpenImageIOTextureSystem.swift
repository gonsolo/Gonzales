import openImageIOBridge

final class OpenImageIOTextureSystem {

        private init() {
                createTextureSystem()
        }

        deinit {
                destroyTextureSystem()
        }

        @MainActor
        private func evaluate(
                filename: String,
                s: Float,
                t: Float,
                pointer: inout UnsafeMutablePointer<Float>
        ) {
                let successful = texture(filename, s, t, pointer)
                if !successful {
                        warnOnce("Could not evaluate texture!")
                }
        }

        @MainActor
        func evaluate(filename: String, s: Float, t: Float) -> FloatX {
                var pointer = UnsafeMutablePointer<Float>.allocate(capacity: 1)
                evaluate(filename: filename, s: s, t: t, pointer: &pointer)
                let result = FloatX(pointer[0])
                return result
        }

        @MainActor
        func evaluate(filename: String, s: Float, t: Float) -> RgbSpectrum {
                var pointer = UnsafeMutablePointer<Float>.allocate(capacity: 3)
                evaluate(filename: filename, s: s, t: t, pointer: &pointer)
                let result = RgbSpectrum(r: pointer[0], g: pointer[1], b: pointer[2])
                return result
        }

        @MainActor
        static let shared = OpenImageIOTextureSystem()
}
