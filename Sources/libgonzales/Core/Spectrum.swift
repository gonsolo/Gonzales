import Foundation

protocol Spectrum: Sendable {
        static func * (lhs: Self, rhs: Self) -> Self

        // Temporary: We adopt a step-by-step strategy to convert the renderer to spectral
        // rendering; first, convert the old Spectrum class to RgbSpectrum (done), then make
        // Spectrum a protocol (done), add additional classes like PiecewiseLinearSpectrum
        // (done) and facilities to convert to RgbSpectrum (here). After that a SampledSpectrum
        // class can be added and all computation converted to this one.
        func asRgb() -> RgbSpectrum
}

struct PiecewiseLinearSpectrum: Spectrum {

        private func findIndex(wavelength: Real) -> Int {
                for (index, lambda) in lambdas.enumerated() where wavelength < lambda {
                        return index
                }
                return lambdas.count - 1
        }

        func asRgb() -> RgbSpectrum {
                let red = values[findIndex(wavelength: 630)]
                let green = values[findIndex(wavelength: 532)]
                let blue = values[findIndex(wavelength: 465)]
                return RgbSpectrum(red: red, green: green, blue: blue)
        }

        let lambdas: [Real]
        let values: [Real]
}

extension PiecewiseLinearSpectrum {
        static func * (_: Self, _: Self) -> Self {
                // This multiplication is not meaningful for piecewise linear spectra
                // and should never be called in practice
                preconditionFailure("PiecewiseLinearSpectrum multiplication not supported")
        }
}

let metalWavelengths: [Real] = [
        298.757050, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835,
        326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509,
        359.374420, 364.659332, 370.102020, 375.709625, 381.489777, 387.450562, 393.600555,
        399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629,
        450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861,
        516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818,
        604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018,
        729.318665, 751.419250, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257,
]

// Silver refractive indices
let silverRefractiveIndices = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
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

let aluminiumRefractiveIndices = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                0.273375, 0.280000, 0.286813, 0.294000, 0.301875, 0.310000, 0.317875,
                0.326000, 0.334750, 0.344000, 0.353813, 0.364000, 0.374375, 0.385000,
                0.395750, 0.407000, 0.419125, 0.432000, 0.445688, 0.460000, 0.474688,
                0.490000, 0.506188, 0.523000, 0.540063, 0.558000, 0.577313, 0.598000,
                0.620313, 0.644000, 0.668625, 0.695000, 0.723750, 0.755000, 0.789000,
                0.826000, 0.867000, 0.912000, 0.963000, 1.020000, 1.080000, 1.150000,
                1.220000, 1.300000, 1.390000, 1.490000, 1.600000, 1.740000, 1.910000,
                2.140000, 2.410000, 2.630000, 2.800000, 2.740000, 2.580000, 2.240000,
        ])

// Copper refractive indices
let copperRefractiveIndices = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.400313, 1.380000, 1.358438, 1.340000, 1.329063, 1.325000, 1.332500,
                1.340000, 1.334375, 1.325000, 1.317812, 1.310000, 1.300313, 1.290000,
                1.281563, 1.270000, 1.249062, 1.225000, 1.200000, 1.180000, 1.174375,
                1.175000, 1.177500, 1.180000, 1.178125, 1.175000, 1.172812, 1.170000,
                1.165312, 1.160000, 1.155312, 1.150000, 1.142812, 1.135000, 1.131562,
                1.120000, 1.092437, 1.040000, 0.950375, 0.826000, 0.645875, 0.468000,
                0.351250, 0.272000, 0.230813, 0.214000, 0.209250, 0.213000, 0.216250,
                0.223000, 0.236500, 0.250000, 0.254188, 0.260000, 0.280000, 0.300000,
        ])

// Silver extinction coefficients
let silverExtinctionCoefficients = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.080000, 0.882000, 0.761063, 0.647000, 0.550875, 0.504000, 0.554375,
                0.663000, 0.818563, 0.986000, 1.120687, 1.240000, 1.345250, 1.440000,
                1.533750, 1.610000, 1.641875, 1.670000, 1.735000, 1.810000, 1.878750,
                1.950000, 2.029375, 2.110000, 2.186250, 2.260000, 2.329375, 2.400000,
                2.478750, 2.560000, 2.640000, 2.720000, 2.798125, 2.880000, 2.973750,
                3.070000, 3.159375, 3.250000, 3.348125, 3.450000, 3.553750, 3.660000,
                3.766250, 3.880000, 4.010625, 4.150000, 4.293125, 4.440000, 4.586250,
                4.740000, 4.908125, 5.090000, 5.288750, 5.500000, 5.720624, 5.950000,
        ])

let aluminiumExtinctionCoefficients = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                3.593750, 3.640000, 3.689375, 3.740000, 3.789375, 3.840000, 3.894375,
                3.950000, 4.005000, 4.060000, 4.113750, 4.170000, 4.233750, 4.300000,
                4.365000, 4.430000, 4.493750, 4.560000, 4.633750, 4.710000, 4.784375,
                4.860000, 4.938125, 5.020000, 5.108750, 5.200000, 5.290000, 5.380000,
                5.480000, 5.580000, 5.690000, 5.800000, 5.915000, 6.030000, 6.150000,
                6.280000, 6.420000, 6.550000, 6.700000, 6.850000, 7.000000, 7.150000,
                7.310000, 7.480000, 7.650000, 7.820000, 8.010000, 8.210000, 8.390000,
                8.570000, 8.620000, 8.600000, 8.450000, 8.310000, 8.210000, 8.210000,
        ])

// Copper extinction coefficients
let copperExtinctionCoefficients = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.662125, 1.687000, 1.703313, 1.720000, 1.744563, 1.770000, 1.791625,
                1.810000, 1.822125, 1.834000, 1.851750, 1.872000, 1.894250, 1.916000,
                1.931688, 1.950000, 1.972438, 2.015000, 2.121562, 2.210000, 2.177188,
                2.130000, 2.160063, 2.210000, 2.249938, 2.289000, 2.326000, 2.362000,
                2.397625, 2.433000, 2.469187, 2.504000, 2.535875, 2.564000, 2.589625,
                2.605000, 2.595562, 2.583000, 2.576500, 2.599000, 2.678062, 2.809000,
                3.010750, 3.240000, 3.458187, 3.670000, 3.863125, 4.050000, 4.239563,
                4.430000, 4.619563, 4.817000, 5.034125, 5.260000, 5.485625, 5.717000,
        ])

let brassWavelengths: [Real] = [
        290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
        430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
        570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700,
        710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840,
        850, 860, 870, 880, 890
]

let brassRefractiveIndices = PiecewiseLinearSpectrum(
        lambdas: brassWavelengths,
        values: [
                1.358, 1.388, 1.419, 1.446, 1.473, 1.494, 1.504, 1.503, 1.497, 1.487, 1.471, 1.445, 1.405, 1.350,
                1.278, 1.191, 1.094, 0.994, 0.900, 0.816, 0.745, 0.686, 0.639, 0.602, 0.573, 0.549, 0.527, 0.505,
                0.484, 0.468, 0.460, 0.450, 0.452, 0.449, 0.445, 0.444, 0.444, 0.445, 0.444, 0.444, 0.445, 0.446,
                0.448, 0.450, 0.452, 0.455, 0.457, 0.458, 0.460, 0.464, 0.469, 0.473, 0.478, 0.481, 0.483, 0.486,
                0.490, 0.494, 0.500, 0.507, 0.515
        ]
)

let brassExtinctionCoefficients = PiecewiseLinearSpectrum(
        lambdas: brassWavelengths,
        values: [
                1.688, 1.731, 1.764, 1.789, 1.807, 1.815, 1.815, 1.815, 1.818, 1.818, 1.813, 1.805, 1.794, 1.786,
                1.784, 1.797, 1.829, 1.883, 1.957, 2.046, 2.145, 2.250, 2.358, 2.464, 2.568, 2.668, 2.765, 2.860,
                2.958, 3.059, 3.159, 3.253, 3.345, 3.434, 3.522, 3.609, 3.695, 3.778, 3.860, 3.943, 4.025, 4.106,
                4.186, 4.266, 4.346, 4.424, 4.501, 4.579, 4.657, 4.737, 4.814, 4.890, 4.965, 5.039, 5.115, 5.192,
                5.269, 5.346, 5.423, 5.500, 5.575
        ]
)

let goldWavelengths: [Real] = [
        298.75705, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835, 326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509, 359.37442, 364.659332, 370.10202, 375.709625, 381.489777, 387.450562, 393.600555, 399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629, 450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861, 516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818, 604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018, 729.318665, 751.41925, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257
]
let goldRefractiveIndices = PiecewiseLinearSpectrum(
        lambdas: goldWavelengths,
        values: [
                1.795, 1.812, 1.822625, 1.83, 1.837125, 1.84, 1.83425, 1.824, 1.812, 1.798, 1.782, 1.766, 1.7525, 1.74, 1.727625, 1.716, 1.705875, 1.696, 1.68475, 1.674, 1.666, 1.658, 1.64725, 1.636, 1.628, 1.616, 1.59625, 1.562, 1.502125, 1.426, 1.345875, 1.242, 1.08675, 0.916, 0.7545, 0.608, 0.49175, 0.402, 0.3455, 0.306, 0.267625, 0.236, 0.212375, 0.194, 0.17775, 0.166, 0.161, 0.16, 0.160875, 0.164, 0.1695, 0.176, 0.181375, 0.188, 0.198125, 0.21
        ]
)
let goldExtinctionCoefficients = PiecewiseLinearSpectrum(
        lambdas: goldWavelengths,
        values: [
                1.920375, 1.92, 1.918875, 1.916, 1.911375, 1.904, 1.891375, 1.878, 1.86825, 1.86, 1.85175, 1.846, 1.84525, 1.848, 1.852375, 1.862, 1.883, 1.906, 1.9225, 1.936, 1.94775, 1.956, 1.959375, 1.958, 1.951375, 1.94, 1.9245, 1.904, 1.875875, 1.846, 1.814625, 1.796, 1.797375, 1.84, 1.9565, 2.12, 2.32625, 2.54, 2.730625, 2.88, 2.940625, 2.97, 3.015, 3.06, 3.07, 3.15, 3.445812, 3.8, 4.087687, 4.357, 4.610188, 4.86, 5.125813, 5.39, 5.63125, 5.88
        ]
)

let namedSpectra: [String: any Spectrum] = [
        "metal-Ag-eta": silverRefractiveIndices,
        "metal-Ag-k": silverExtinctionCoefficients,
        "metal-Al-eta": aluminiumRefractiveIndices,
        "metal-Al-k": aluminiumExtinctionCoefficients,
        "metal-Cu-eta": copperRefractiveIndices,
        "metal-Cu-k": copperExtinctionCoefficients,
        "metal-CuZn-eta": brassRefractiveIndices,
        "metal-CuZn-k": brassExtinctionCoefficients,
        "metal-Au-eta": goldRefractiveIndices,
        "metal-Au-k": goldExtinctionCoefficients,
]

public struct RgbSpectrum: Initializable, Sendable, ThreeComponent, Spectrum {

        init() {
                self.init(red: 0, green: 0, blue: 0)
        }

        public init(x: Real, y: Real, z: Real) {
                self.xyz = SIMD4<Real>(x, y, z, 1.0)
        }

        public init(red: Real, green: Real, blue: Real) {
                self.init(x: red, y: green, z: blue)
        }

        init(intensity: Real) {
                self.init(x: intensity, y: intensity, z: intensity)
        }

        init(rgb: (FloatType, FloatType, FloatType)) {
                self.init(x: rgb.0, y: rgb.1, z: rgb.2)
        }

        var isNaN: Bool {
                return red.isNaN || green.isNaN || blue.isNaN
        }

        var isInfinite: Bool {
                return red.isInfinite || green.isInfinite || blue.isInfinite
        }

        // Convenience accessors
        var red: FloatType {
                get { return x }
                set { x = newValue }
        }

        var green: FloatType {
                get { return y }
                set { y = newValue }
        }

        var blue: FloatType {
                get { return z }
                set { z = newValue }
        }

        public var x: Real {
                get { return xyz.x }
                set { xyz.x = newValue }
        }
        public var y: Real {
                get { return xyz.y }
                set { xyz.y = newValue }
        }
        public var z: Real {
                get { return xyz.z }
                set { xyz.z = newValue }
        }

        var xyz: SIMD4<Real>
}

extension RgbSpectrum: CustomStringConvertible {

        public var description: String {
                return "[ \(red) \(green) \(blue) ]"
        }
}

extension RgbSpectrum {

        var isBlack: Bool {
                return red == 0 && green == 0 && blue == 0
        }
}

extension RgbSpectrum {

        // MARK: - SIMD Arithmetic

        public static func * (mul: Real, spectrum: RgbSpectrum) -> RgbSpectrum {
                var result = RgbSpectrum()
                result.xyz = mul * spectrum.xyz
                return result
        }

        public static func * (spectrum: RgbSpectrum, mul: Real) -> RgbSpectrum {
                return mul * spectrum
        }

        public static func *= (left: inout RgbSpectrum, right: RgbSpectrum) {
                left.xyz *= right.xyz
        }

        public static func + (spectrum: RgbSpectrum, value: Real) -> RgbSpectrum {
                var result = RgbSpectrum()
                result.xyz = spectrum.xyz + value
                return result
        }

        public static func - (spectrum: RgbSpectrum, value: Real) -> RgbSpectrum {
                var result = RgbSpectrum()
                result.xyz = spectrum.xyz - value
                return result
        }

        public static func / (
                numerator: RgbSpectrum,
                denominator: RgbSpectrum
        ) -> RgbSpectrum {
                var result = RgbSpectrum()
                result.xyz = numerator.xyz / denominator.xyz
                return result
        }

        public static func == (lhs: RgbSpectrum, rhs: RgbSpectrum) -> Bool {
                return lhs.red == rhs.red && lhs.green == rhs.green && lhs.blue == rhs.blue
        }

        public static func != (lhs: RgbSpectrum, rhs: RgbSpectrum) -> Bool {
                return !(lhs == rhs)
        }

        public func squareRoot() -> RgbSpectrum {
                var result = RgbSpectrum()
                result.xyz = xyz.squareRoot()
                return result
        }

        func average() -> FloatType {
                return (red + green + blue) / 3
        }

        var maxValue: FloatType {
                return max(red, max(green, blue))
        }

        subscript(index: Int) -> FloatType {
                get {
                        switch index {
                        case 0: return red
                        case 1: return green
                        case 2: return blue
                        default: return red
                        }
                }
                set(newValue) {
                        switch index {
                        case 0: red = newValue
                        case 1: green = newValue
                        case 2: blue = newValue
                        default: red = newValue
                        }
                }
        }

}

extension RgbSpectrum {

        func asRgb() -> RgbSpectrum {
                return self
        }

        var luminance: FloatType {
                let redWeight: FloatType = 0.212671 * red
                let greenWeight: FloatType = 0.715160 * green
                let blueWeight: FloatType = 0.072169 * blue
                return redWeight + greenWeight + blueWeight
        }
}

extension RgbSpectrum {

        init(from normal: Normal3) {
                let normal = normalized(normal)
                self.init(red: abs(normal.x), green: abs(normal.y), blue: abs(normal.z))
        }
}

let black = RgbSpectrum(intensity: 0)
let gray = RgbSpectrum(intensity: 0.5)
let white = RgbSpectrum(intensity: 1)
public let red = RgbSpectrum(red: 1, green: 0, blue: 0)
let blue = RgbSpectrum(red: 0, green: 0, blue: 1)
let green = RgbSpectrum(red: 0, green: 1, blue: 0)

func pow(base: RgbSpectrum, exp: Real) -> RgbSpectrum {
        return RgbSpectrum(
                red: pow(base.red, exp),
                green: pow(base.green, exp),
                blue: pow(base.blue, exp))
}

func exp(_ x: RgbSpectrum) -> RgbSpectrum {
        return RgbSpectrum(
                red: exp(x.red),
                green: exp(x.green),
                blue: exp(x.blue))
}

func gammaLinearToSrgb(light: RgbSpectrum) -> RgbSpectrum {
        var converted = RgbSpectrum()
        converted.red = gammaLinearToSrgb(value: light.red)
        converted.green = gammaLinearToSrgb(value: light.green)
        converted.blue = gammaLinearToSrgb(value: light.blue)
        return converted
}

func gammaSrgbToLinear(light: RgbSpectrum) -> RgbSpectrum {
        var converted = RgbSpectrum()
        converted.red = gammaSrgbToLinear(value: light.red)
        converted.green = gammaSrgbToLinear(value: light.green)
        converted.blue = gammaSrgbToLinear(value: light.blue)
        return converted
}

/// Approximate blackbody radiation as RGB using Tanner Helland's algorithm.
/// Returns normalized [0,1] RGB values for the given color temperature in Kelvin.
func blackBodyToRgb(kelvin: Real) -> RgbSpectrum {
        let temp = kelvin / 100

        let red: Real
        if temp <= 66 {
                red = 1.0
        } else {
                let x = temp - 60
                red = min(1.0, max(0.0, Real(1.29293618606 * pow(Double(x), -0.1332047592))))
        }

        let green: Real
        if temp <= 66 {
                let x = temp
                green = min(1.0, max(0.0, Real(0.39008157876 * log(Double(x)) - 0.63184144378)))
        } else {
                let x = temp - 60
                green = min(1.0, max(0.0, Real(1.12989086090 * pow(Double(x), -0.0755148492))))
        }

        let blue: Real
        if temp >= 66 {
                blue = 1.0
        } else if temp <= 19 {
                blue = 0.0
        } else {
                let x = temp - 10
                blue = min(1.0, max(0.0, Real(0.54320678911 * log(Double(x)) - 1.19625408914)))
        }

        return RgbSpectrum(rgb: (red, green, blue))
}
