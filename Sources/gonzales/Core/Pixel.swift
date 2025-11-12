enum PixelError: Error {
        case zeroWeight
}

struct Pixel {

        init(light: RgbSpectrum = RgbSpectrum(), weight: FloatX = 0) {
                self.light = light
                self.weight = weight
        }

        func normalized() throws -> Pixel {
                if light.isBlack { return Pixel(light: self.light, weight: 1) }
                guard !weight.isZero else { throw PixelError.zeroWeight }
                return Pixel(light: self.light / self.weight, weight: 1)
        }

        func intensity() -> FloatX {
                return (light.red + light.green + light.blue) / 3.0
        }

        var light: RgbSpectrum
        private var weight: FloatX
}

extension Pixel: CustomStringConvertible {
        public var description: String {
                return "[ \(light) \(weight) ]"
        }
}

extension Pixel {
        static func + (left: Pixel, right: Pixel) -> Pixel {
                return Pixel(light: left.light + right.light, weight: left.weight + right.weight)
        }
}
