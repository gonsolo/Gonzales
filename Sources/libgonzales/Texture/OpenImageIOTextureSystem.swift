import openImageIOBridge

final class OpenImageIOTextureSystem: Sendable {

        private init() {
                createTextureSystem()
        }

        deinit {
                destroyTextureSystem()
        }

        private func evaluate(
                filename: String,
                s: Float,
                t: Float,
                pointer: inout UnsafeMutablePointer<Float>
        ) {
                let successful = texture(filename, s, t, pointer)
                if !successful {
                        print("Could not evaluate texture!")
                }
        }

        func evaluate(filename: String, s: Float, t: Float) -> FloatX {
                var pointer = UnsafeMutablePointer<Float>.allocate(capacity: 1)
                evaluate(filename: filename, s: s, t: t, pointer: &pointer)
                let result = FloatX(pointer[0])
                return result
        }

        func evaluate(filename: String, s: Float, t: Float) -> RgbSpectrum {
                var pointer = UnsafeMutablePointer<Float>.allocate(capacity: 3)
                evaluate(filename: filename, s: s, t: t, pointer: &pointer)
                let result = RgbSpectrum(red: pointer[0], green: pointer[1], blue: pointer[2])
                return result
        }

        static let shared = OpenImageIOTextureSystem()
}
