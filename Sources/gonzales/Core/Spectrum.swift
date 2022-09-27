import Foundation

public struct BaseSpectrum<T: FloatingPoint>: Initializable, Three {

        init() {
                self.init(r: 0, g: 0, b: 0)
        }

        init(x: T, y: T, z: T) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(r: T, g: T, b: T) {
                self.init(x: r, y: g, z: b)
        }

        init(intensity: T) {
                self.x = intensity
                self.y = intensity
                self.z = intensity
        }

        init(rgb: (T, T, T)) {
                self.x = rgb.0
                self.y = rgb.1
                self.z = rgb.2
        }

        var isNaN: Bool {
                return r.isNaN || g.isNaN || b.isNaN
        }

        var isInfinite: Bool {
                return r.isInfinite || g.isInfinite || b.isInfinite
        }

        // Conform to Three
        var x: T
        var y: T
        var z: T

        // Convenience accessors for Spectrum
        var r: T {
                get { return x }
                set { x = newValue }
        }

        var g: T {
                get { return y }
                set { y = newValue }
        }

        var b: T {
                get { return z }
                set { z = newValue }
        }
}

extension BaseSpectrum: CustomStringConvertible {

        public var description: String {
                return "[ \(r) \(g) \(b) ]"
        }
}

extension BaseSpectrum {

        var isBlack: Bool {
                return r == 0 && g == 0 && b == 0
        }
}

extension BaseSpectrum where T: FloatingPoint {

        public static func * (mul: T, spectrum: BaseSpectrum<T>) -> BaseSpectrum {
                return BaseSpectrum(r: mul * spectrum.x, g: mul * spectrum.y, b: mul * spectrum.z)
        }

        public static func * (spectrum: BaseSpectrum<T>, mul: T) -> BaseSpectrum {
                return mul * spectrum
        }

        public static func + (spectrum: BaseSpectrum<T>, value: T) -> BaseSpectrum {
                return BaseSpectrum(
                        r: spectrum.r + value,
                        g: spectrum.g + value,
                        b: spectrum.b + value)
        }

        public static func - (spectrum: BaseSpectrum<T>, value: T) -> BaseSpectrum {
                return BaseSpectrum(
                        r: spectrum.r - value,
                        g: spectrum.g - value,
                        b: spectrum.b - value)
        }

        public static func / (
                numerator: BaseSpectrum<T>,
                denominator: BaseSpectrum<T>
        ) -> BaseSpectrum {
                return BaseSpectrum(
                        r: numerator.r / denominator.r,
                        g: numerator.g / denominator.g,
                        b: numerator.b / denominator.b)
        }

        public static func == (a: BaseSpectrum<T>, b: BaseSpectrum<T>) -> Bool {
                return a.r == b.r && a.g == b.g && a.b == b.b
        }

        public static func != (a: BaseSpectrum<T>, b: BaseSpectrum<T>) -> Bool {
                return !(a == b)
        }

        public func squareRoot() -> BaseSpectrum {
                return BaseSpectrum(
                        r: r.squareRoot(),
                        g: g.squareRoot(),
                        b: b.squareRoot())
        }

        func average() -> T {
                return (r + g + b) / 3
        }
}

extension BaseSpectrum where T: BinaryFloatingPoint {

        var luminance: T {
                let rw: T = 0.212671 * r
                let gw: T = 0.715160 * g
                let bw: T = 0.072169 * b
                return rw + gw + bw
        }
}

extension BaseSpectrum where T: FloatingPoint {

        init(from normal: Normal3<T>) {
                let normal = normalized(normal)
                self.init(r: abs(normal.x), g: abs(normal.y), b: abs(normal.z))
        }
}

typealias Spectrum = BaseSpectrum<FloatX>

let black = Spectrum(intensity: 0)
let gray = Spectrum(intensity: 0.5)
let white = Spectrum(intensity: 1)
let red = Spectrum(r: 1, g: 0, b: 0)
let blue = Spectrum(r: 0, g: 0, b: 1)
let green = Spectrum(r: 0, g: 1, b: 0)

func pow(base: Spectrum, exp: FloatX) -> Spectrum {
        return Spectrum(
                r: pow(base.r, exp),
                g: pow(base.g, exp),
                b: pow(base.b, exp))
}

func gammaLinearToSrgb(light: Spectrum) -> Spectrum {
        var converted = Spectrum()
        converted.r = gammaLinearToSrgb(value: light.r)
        converted.g = gammaLinearToSrgb(value: light.g)
        converted.b = gammaLinearToSrgb(value: light.b)
        return converted
}

func gammaSrgbToLinear(light: Spectrum) -> Spectrum {
        var converted = Spectrum()
        converted.r = gammaSrgbToLinear(value: light.r)
        converted.g = gammaSrgbToLinear(value: light.g)
        converted.b = gammaSrgbToLinear(value: light.b)
        return converted
}
