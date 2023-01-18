import Foundation

protocol Spectrum {}

public struct BaseRGBSpectrum<T: FloatingPoint>: Initializable, Spectrum, Three {

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

        // Convenience accessors for RGBSpectrum
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

extension BaseRGBSpectrum: CustomStringConvertible {

        public var description: String {
                return "[ \(r) \(g) \(b) ]"
        }
}

extension BaseRGBSpectrum {

        var isBlack: Bool {
                return r == 0 && g == 0 && b == 0
        }
}

extension BaseRGBSpectrum where T: FloatingPoint {

        public static func * (mul: T, spectrum: BaseRGBSpectrum<T>) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(r: mul * spectrum.x, g: mul * spectrum.y, b: mul * spectrum.z)
        }

        public static func * (spectrum: BaseRGBSpectrum<T>, mul: T) -> BaseRGBSpectrum {
                return mul * spectrum
        }

        public static func + (spectrum: BaseRGBSpectrum<T>, value: T) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: spectrum.r + value,
                        g: spectrum.g + value,
                        b: spectrum.b + value)
        }

        public static func - (spectrum: BaseRGBSpectrum<T>, value: T) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: spectrum.r - value,
                        g: spectrum.g - value,
                        b: spectrum.b - value)
        }

        public static func / (
                numerator: BaseRGBSpectrum<T>,
                denominator: BaseRGBSpectrum<T>
        ) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: numerator.r / denominator.r,
                        g: numerator.g / denominator.g,
                        b: numerator.b / denominator.b)
        }

        public static func == (a: BaseRGBSpectrum<T>, b: BaseRGBSpectrum<T>) -> Bool {
                return a.r == b.r && a.g == b.g && a.b == b.b
        }

        public static func != (a: BaseRGBSpectrum<T>, b: BaseRGBSpectrum<T>) -> Bool {
                return !(a == b)
        }

        public func squareRoot() -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: r.squareRoot(),
                        g: g.squareRoot(),
                        b: b.squareRoot())
        }

        func average() -> T {
                return (r + g + b) / 3
        }
}

extension BaseRGBSpectrum where T: BinaryFloatingPoint {

        var luminance: T {
                let rw: T = 0.212671 * r
                let gw: T = 0.715160 * g
                let bw: T = 0.072169 * b
                return rw + gw + bw
        }
}

extension BaseRGBSpectrum where T: FloatingPoint {

        init(from normal: Normal3<T>) {
                let normal = normalized(normal)
                self.init(r: abs(normal.x), g: abs(normal.y), b: abs(normal.z))
        }
}

typealias RGBSpectrum = BaseRGBSpectrum<FloatX>

let black = RGBSpectrum(intensity: 0)
let gray = RGBSpectrum(intensity: 0.5)
let white = RGBSpectrum(intensity: 1)
let red = RGBSpectrum(r: 1, g: 0, b: 0)
let blue = RGBSpectrum(r: 0, g: 0, b: 1)
let green = RGBSpectrum(r: 0, g: 1, b: 0)

func pow(base: RGBSpectrum, exp: FloatX) -> RGBSpectrum {
        return RGBSpectrum(
                r: pow(base.r, exp),
                g: pow(base.g, exp),
                b: pow(base.b, exp))
}

func gammaLinearToSrgb(light: RGBSpectrum) -> RGBSpectrum {
        var converted = RGBSpectrum()
        converted.r = gammaLinearToSrgb(value: light.r)
        converted.g = gammaLinearToSrgb(value: light.g)
        converted.b = gammaLinearToSrgb(value: light.b)
        return converted
}

func gammaSrgbToLinear(light: RGBSpectrum) -> RGBSpectrum {
        var converted = RGBSpectrum()
        converted.r = gammaSrgbToLinear(value: light.r)
        converted.g = gammaSrgbToLinear(value: light.g)
        converted.b = gammaSrgbToLinear(value: light.b)
        return converted
}
