import XCTest

@testable import libgonzales

final class SpectrumTests: XCTestCase {

        // MARK: - Scalar Multiplication

        func testScalarMultiplication() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = 2.0 * spectrum
                XCTAssertEqual(result.red, 2.0, accuracy: 1e-6)
                XCTAssertEqual(result.green, 4.0, accuracy: 1e-6)
                XCTAssertEqual(result.blue, 6.0, accuracy: 1e-6)
        }

        func testScalarMultiplicationCommutative() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let left = 2.0 * spectrum
                let right = spectrum * 2.0
                XCTAssertEqual(left.red, right.red, accuracy: 1e-6)
                XCTAssertEqual(left.green, right.green, accuracy: 1e-6)
                XCTAssertEqual(left.blue, right.blue, accuracy: 1e-6)
        }

        // MARK: - Spectrum-Spectrum Multiplication

        func testSpectrumMultiplication() {
                var a = RgbSpectrum(red: 2, green: 3, blue: 4)
                let b = RgbSpectrum(red: 0.5, green: 0.5, blue: 0.5)
                a *= b
                XCTAssertEqual(a.red, 1.0, accuracy: 1e-6)
                XCTAssertEqual(a.green, 1.5, accuracy: 1e-6)
                XCTAssertEqual(a.blue, 2.0, accuracy: 1e-6)
        }

        func testMultiplyByWhiteIsIdentity() {
                let original = RgbSpectrum(red: 0.3, green: 0.6, blue: 0.9)
                var result = original
                result *= white
                XCTAssertEqual(result.red, original.red, accuracy: 1e-6)
                XCTAssertEqual(result.green, original.green, accuracy: 1e-6)
                XCTAssertEqual(result.blue, original.blue, accuracy: 1e-6)
        }

        func testMultiplyByBlackIsBlack() {
                var spectrum = RgbSpectrum(red: 0.5, green: 0.5, blue: 0.5)
                spectrum *= black
                XCTAssertTrue(spectrum.isBlack)
        }

        // MARK: - Addition and Subtraction

        func testAddScalar() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = spectrum + 0.5
                XCTAssertEqual(result.red, 1.5, accuracy: 1e-6)
                XCTAssertEqual(result.green, 2.5, accuracy: 1e-6)
                XCTAssertEqual(result.blue, 3.5, accuracy: 1e-6)
        }

        func testSubtractScalar() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                let result = spectrum - 0.5
                XCTAssertEqual(result.red, 0.5, accuracy: 1e-6)
                XCTAssertEqual(result.green, 1.5, accuracy: 1e-6)
                XCTAssertEqual(result.blue, 2.5, accuracy: 1e-6)
        }

        // MARK: - Division

        func testDivision() {
                let a = RgbSpectrum(red: 4, green: 9, blue: 16)
                let b = RgbSpectrum(red: 2, green: 3, blue: 4)
                let result = a / b
                XCTAssertEqual(result.red, 2.0, accuracy: 1e-6)
                XCTAssertEqual(result.green, 3.0, accuracy: 1e-6)
                XCTAssertEqual(result.blue, 4.0, accuracy: 1e-6)
        }

        // MARK: - Edge Cases

        func testIsBlack() {
                XCTAssertTrue(black.isBlack)
                XCTAssertFalse(white.isBlack)
                XCTAssertFalse(RgbSpectrum(red: 0.001, green: 0, blue: 0).isBlack)
        }

        func testIsNaN() {
                let nanSpectrum = RgbSpectrum(red: FloatX.nan, green: 0, blue: 0)
                XCTAssertTrue(nanSpectrum.isNaN)
                XCTAssertFalse(white.isNaN)
        }

        func testIsInfinite() {
                let infSpectrum = RgbSpectrum(red: FloatX.infinity, green: 0, blue: 0)
                XCTAssertTrue(infSpectrum.isInfinite)
                XCTAssertFalse(white.isInfinite)
        }

        // MARK: - Derived Values

        func testLuminance() {
                // ITU-R BT.709 weights: 0.212671, 0.715160, 0.072169
                let spectrum = RgbSpectrum(red: 1, green: 1, blue: 1)
                let expectedLuminance: FloatX = 0.212671 + 0.715160 + 0.072169
                XCTAssertEqual(spectrum.luminance, expectedLuminance, accuracy: 1e-4)
        }

        func testLuminanceRedOnly() {
                let redOnly = RgbSpectrum(red: 1, green: 0, blue: 0)
                XCTAssertEqual(redOnly.luminance, 0.212671, accuracy: 1e-4)
        }

        func testMaxValue() {
                let spectrum = RgbSpectrum(red: 0.3, green: 0.9, blue: 0.1)
                XCTAssertEqual(spectrum.maxValue, 0.9, accuracy: 1e-6)
        }

        func testAverage() {
                let spectrum = RgbSpectrum(red: 3, green: 6, blue: 9)
                XCTAssertEqual(spectrum.average(), 6.0, accuracy: 1e-6)
        }

        func testSquareRoot() {
                let spectrum = RgbSpectrum(red: 4, green: 9, blue: 16)
                let result = spectrum.squareRoot()
                XCTAssertEqual(result.red, 2.0, accuracy: 1e-6)
                XCTAssertEqual(result.green, 3.0, accuracy: 1e-6)
                XCTAssertEqual(result.blue, 4.0, accuracy: 1e-6)
        }

        // MARK: - Equality

        func testEquality() {
                let a = RgbSpectrum(red: 1, green: 2, blue: 3)
                let b = RgbSpectrum(red: 1, green: 2, blue: 3)
                XCTAssertTrue(a == b)
        }

        func testInequality() {
                let a = RgbSpectrum(red: 1, green: 2, blue: 3)
                let b = RgbSpectrum(red: 1, green: 2, blue: 4)
                XCTAssertTrue(a != b)
        }

        // MARK: - Subscript

        func testSubscriptGet() {
                let spectrum = RgbSpectrum(red: 1, green: 2, blue: 3)
                XCTAssertEqual(spectrum[0], 1.0, accuracy: 1e-6)
                XCTAssertEqual(spectrum[1], 2.0, accuracy: 1e-6)
                XCTAssertEqual(spectrum[2], 3.0, accuracy: 1e-6)
        }

        func testSubscriptSet() {
                var spectrum = RgbSpectrum()
                spectrum[0] = 10
                spectrum[1] = 20
                spectrum[2] = 30
                XCTAssertEqual(spectrum.red, 10, accuracy: 1e-6)
                XCTAssertEqual(spectrum.green, 20, accuracy: 1e-6)
                XCTAssertEqual(spectrum.blue, 30, accuracy: 1e-6)
        }

        // MARK: - Gamma Conversion

        func testGammaLinearToSrgbLow() {
                // Below threshold 0.0031308
                let value: FloatX = 0.001
                let result = gammaLinearToSrgb(value: value)
                XCTAssertEqual(result, value * 12.92, accuracy: 1e-6)
        }

        func testGammaLinearToSrgbHigh() {
                // Above threshold
                let value: FloatX = 0.5
                let result = gammaLinearToSrgb(value: value)
                let expected: FloatX = 1.055 * pow(value, 1.0 / 2.4) - 0.055
                XCTAssertEqual(result, expected, accuracy: 1e-5)
        }

        func testGammaRoundTrip() {
                let original: FloatX = 0.5
                let srgb = gammaLinearToSrgb(value: original)
                let linear = gammaSrgbToLinear(value: srgb)
                XCTAssertEqual(linear, original, accuracy: 1e-4)
        }
}
