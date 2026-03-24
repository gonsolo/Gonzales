import Testing

@testable import libgonzales

@Suite struct PixelTests {

        @Test func pixelNormalizedDividesLightByWeight() throws {
                let pixel = Pixel(light: RgbSpectrum(red: 4, green: 6, blue: 8), weight: 2)
                let result = try pixel.normalized()
                #expect(abs(result.light.red - 2.0) <= 1e-6)
                #expect(abs(result.light.green - 3.0) <= 1e-6)
                #expect(abs(result.light.blue - 4.0) <= 1e-6)
        }

        @Test func pixelBlackNormalizesWithoutThrow() throws {
                let pixel = Pixel(light: black, weight: 0)
                let result = try pixel.normalized()
                #expect(result.light.isBlack)
        }

        @Test func pixelZeroWeightNonBlackThrows() {
                let pixel = Pixel(light: white, weight: 0)
                #expect(throws: PixelError.self) {
                        try pixel.normalized()
                }
        }

        @Test func pixelIntensityIsAverage() {
                let pixel = Pixel(light: RgbSpectrum(red: 3, green: 6, blue: 9))
                #expect(abs(pixel.intensity() - 6.0) <= 1e-6)
        }

        @Test func pixelAddition() {
                let a = Pixel(light: RgbSpectrum(red: 1, green: 2, blue: 3), weight: 1)
                let b = Pixel(light: RgbSpectrum(red: 4, green: 5, blue: 6), weight: 2)
                let result = a + b
                #expect(abs(result.light.red - 5) <= 1e-6)
                #expect(abs(result.light.green - 7) <= 1e-6)
                #expect(abs(result.light.blue - 9) <= 1e-6)
        }

        @Test func pixelPlusEquals() {
                var pixel = Pixel(light: RgbSpectrum(red: 1, green: 1, blue: 1), weight: 1)
                pixel += Pixel(light: RgbSpectrum(red: 2, green: 3, blue: 4), weight: 1)
                #expect(abs(pixel.light.red - 3) <= 1e-6)
                #expect(abs(pixel.light.green - 4) <= 1e-6)
                #expect(abs(pixel.light.blue - 5) <= 1e-6)
        }
}
