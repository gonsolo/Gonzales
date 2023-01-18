import Foundation

protocol Spectrum {}

struct PiecewiseLinearSpectrum: Spectrum {

        let lambdas: [FloatX]
        let values: [FloatX]
}

let ageta = PiecewiseLinearSpectrum(
        lambdas: [
                298.757050, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835,
                326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509,
                359.374420, 364.659332, 370.102020, 375.709625, 381.489777, 387.450562, 393.600555,
                399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629,
                450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861,
                516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818,
                604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018,
                729.318665, 751.419250, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257,
        ],
        values: [
                1.519000, 1.496000, 1.432500, 1.323000, 1.142062, 0.932000, 0.719062,
                0.526000, 0.388125, 0.294000, 0.253313, 0.238000, 0.221438, 0.209000,
                0.194813, 0.186000, 0.192063, 0.200000, 0.198063, 0.192000, 0.182000,
                0.173000, 0.172625, 0.173000, 0.166688, 0.160000, 0.158500, 0.157000,
                0.151063, 0.144000, 0.137313, 0.132000, 0.130250, 0.130000, 0.129938,
                0.130000, 0.130063, 0.129000, 0.124375, 0.120000, 0.119313, 0.121000,
                0.125500, 0.131000, 0.136125, 0.140000, 0.140063, 0.140000, 0.144313,
                0.148000, 0.145875, 0.143000, 0.142563, 0.145000, 0.151938, 0.163000,
        ])

var namedSpectra: [String: Spectrum] = [
        "metal-Ag-eta": ageta
]

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
                return BaseRGBSpectrum(
                        r: mul * spectrum.x,
                        g: mul * spectrum.y,
                        b: mul * spectrum.z)
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
