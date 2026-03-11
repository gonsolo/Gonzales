import Glibc
import Testing

@testable import libgonzales

@Suite struct SpectrumTests {

        // MARK: - Scalar Multiplication

        @Test func scalarMultiplication() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = 2.0 * spectrum
                #expect(abs(result.red - 2.0) <= 1e-6)
                #expect(abs(result.green - 4.0) <= 1e-6)
                #expect(abs(result.blue - 6.0) <= 1e-6)
        }

        @Test func scalarMultiplicationCommutative() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let left = 2.0 * spectrum
                let right = spectrum * 2.0
                #expect(abs(left.red - right.red) <= 1e-6)
                #expect(abs(left.green - right.green) <= 1e-6)
                #expect(abs(left.blue - right.blue) <= 1e-6)
        }

        // MARK: - Spectrum-Spectrum Multiplication

        @Test func spectrumMultiplication() {
                var a = RgbSpectrum(red: 2, green: 3, blue: 4)
                let b = RgbSpectrum(red: 0.5, green: 0.5, blue: 0.5)
                a *= b
                #expect(abs(a.red - 1.0) <= 1e-6)
                #expect(abs(a.green - 1.5) <= 1e-6)
                #expect(abs(a.blue - 2.0) <= 1e-6)
        }

        @Test func multiplyByWhiteIsIdentity() {
                let original = RgbSpectrum(red: 0.3, green: 0.6, blue: 0.9)
                var result = original
                result *= white
                #expect(abs(result.red - original.red) <= 1e-6)
                #expect(abs(result.green - original.green) <= 1e-6)
                #expect(abs(result.blue - original.blue) <= 1e-6)
        }

        @Test func multiplyByBlackIsBlack() {
                var spectrum = RgbSpectrum(red: 0.5, green: 0.5, blue: 0.5)
                spectrum *= black
                #expect(spectrum.isBlack)
        }

        // MARK: - Addition and Subtraction

        @Test func addScalar() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = spectrum + 0.5
                #expect(abs(result.red - 1.5) <= 1e-6)
                #expect(abs(result.green - 2.5) <= 1e-6)
                #expect(abs(result.blue - 3.5) <= 1e-6)
        }

        @Test func subtractScalar() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = spectrum - 0.5
                #expect(abs(result.red - 0.5) <= 1e-6)
                #expect(abs(result.green - 1.5) <= 1e-6)
                #expect(abs(result.blue - 2.5) <= 1e-6)
        }

        // MARK: - Division

        @Test func division() {
                let a = RgbSpectrum(red: 4, green: 9, blue: 16)
                let b = RgbSpectrum(red: 2, green: 3, blue: 4)
                let result = a / b
                #expect(abs(result.red - 2.0) <= 1e-6)
                #expect(abs(result.green - 3.0) <= 1e-6)
                #expect(abs(result.blue - 4.0) <= 1e-6)
        }

        // MARK: - Edge Cases

        @Test func isBlack() {
                #expect(black.isBlack)
                #expect(!white.isBlack)
                #expect(!RgbSpectrum(red: 0.001, green: 0, blue: 0).isBlack)
        }

        @Test func isNaN() {
                let nanSpectrum = RgbSpectrum(red: FloatX.nan, green: 0, blue: 0)
                #expect(nanSpectrum.isNaN)
                #expect(!white.isNaN)
        }

        @Test func isInfinite() {
                let infSpectrum = RgbSpectrum(red: FloatX.infinity, green: 0, blue: 0)
                #expect(infSpectrum.isInfinite)
                #expect(!white.isInfinite)
        }

        // MARK: - Derived Values

        @Test func luminance() {
                // ITU-R BT.709 weights: 0.212671, 0.715160, 0.072169
                let spectrum = RgbSpectrum(red: 1, green: 1, blue: 1)
                let expectedLuminance: FloatX = 0.212671 + 0.715160 + 0.072169
                #expect(abs(spectrum.luminance - expectedLuminance) <= 1e-4)
        }

        @Test func luminanceRedOnly() {
                let redOnly = RgbSpectrum(red: 1, green: 0, blue: 0)
                #expect(abs(redOnly.luminance - 0.212671) <= 1e-4)
        }

        @Test func maxValue() {
                let spectrum = RgbSpectrum(red: 0.3, green: 0.9, blue: 0.1)
                #expect(abs(spectrum.maxValue - 0.9) <= 1e-6)
        }

        @Test func average() {
                let spectrum = RgbSpectrum(red: 3, green: 6, blue: 9)
                #expect(abs(spectrum.average() - 6.0) <= 1e-6)
        }

        @Test func squareRoot() {
                let spectrum = RgbSpectrum(red: 4, green: 9, blue: 16)
                let result = spectrum.squareRoot()
                #expect(abs(result.red - 2.0) <= 1e-6)
                #expect(abs(result.green - 3.0) <= 1e-6)
                #expect(abs(result.blue - 4.0) <= 1e-6)
        }

        // MARK: - Equality

        @Test func equality() {
                let a = RgbSpectrum(red: 1, green: 2, blue: 3)
                let b = RgbSpectrum(red: 1, green: 2, blue: 3)
                #expect(a == b)
        }

        @Test func inequality() {
                let a = RgbSpectrum(red: 1, green: 2, blue: 3)
                let b = RgbSpectrum(red: 1, green: 2, blue: 4)
                #expect(a != b)
        }

        // MARK: - Subscript

        @Test func subscriptGet() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                #expect(abs(spectrum[0] - 1.0) <= 1e-6)
                #expect(abs(spectrum[1] - 2.0) <= 1e-6)
                #expect(abs(spectrum[2] - 3.0) <= 1e-6)
        }

        @Test func subscriptSet() {
                var spectrum = RgbSpectrum()
                spectrum[0] = 10
                spectrum[1] = 20
                spectrum[2] = 30
                #expect(abs(spectrum.red - 10) <= 1e-6)
                #expect(abs(spectrum.green - 20) <= 1e-6)
                #expect(abs(spectrum.blue - 30) <= 1e-6)
        }

        // MARK: - Gamma Conversion

        @Test func gammaLinearToSrgbLow() {
                // Below threshold 0.0031308
                let value: FloatX = 0.001
                let result = gammaLinearToSrgb(value: value)
                #expect(abs(result - value * 12.92) <= 1e-6)
        }

        @Test func gammaLinearToSrgbHigh() {
                // Above threshold
                let value: FloatX = 0.5
                let result = gammaLinearToSrgb(value: value)
                let expected: FloatX = 1.055 * Glibc.powf(value, 1.0 / 2.4) - 0.055
                #expect(abs(result - expected) <= 1e-5)
        }

        @Test func gammaRoundTrip() {
                let original: FloatX = 0.5
                let srgb = gammaLinearToSrgb(value: original)
                let linear = gammaSrgbToLinear(value: srgb)
                #expect(abs(linear - original) <= 1e-4)
        }
}
