import Testing

@testable import libgonzales

@Suite struct BlackBodyTests {

        @Test func blackBody6500KIsWhitish() {
                let rgb = blackBodyToRgb(kelvin: 6500)
                // Daylight ~6500K should be near white
                #expect(rgb.red > 0.9, "Red should be near 1.0 at 6500K")
                #expect(rgb.green > 0.9, "Green should be near 1.0 at 6500K")
                #expect(rgb.blue > 0.9, "Blue should be near 1.0 at 6500K")
        }

        @Test func blackBody2000KIsReddish() {
                let rgb = blackBodyToRgb(kelvin: 2000)
                // Candlelight is warm/red
                #expect(rgb.red > rgb.green, "Red > green at 2000K")
                #expect(rgb.green > rgb.blue, "Green > blue at 2000K")
        }

        @Test func blackBody10000KIsBluish() {
                let rgb = blackBodyToRgb(kelvin: 10000)
                // Very hot = bluish
                #expect(abs(rgb.blue - 1.0) <= 1e-3, "Blue should be 1.0 at 10000K")
                #expect(rgb.red < 1.0, "Red should be < 1.0 at 10000K")
        }

        @Test func blackBodyChannelsClampedToZeroOne() {
                let temperatures: [Real] = [1000, 2000, 3000, 5000, 6500, 8000, 10000, 15000, 20000]
                for kelvin in temperatures {
                        let rgb = blackBodyToRgb(kelvin: kelvin)
                        #expect(rgb.red >= 0 && rgb.red <= 1, "Red out of [0,1] at \(kelvin)K")
                        #expect(rgb.green >= 0 && rgb.green <= 1, "Green out of [0,1] at \(kelvin)K")
                        #expect(rgb.blue >= 0 && rgb.blue <= 1, "Blue out of [0,1] at \(kelvin)K")
                }
        }

        @Test func blackBodyVeryLowTempIsRedOnly() {
                let rgb = blackBodyToRgb(kelvin: 1000)
                #expect(rgb.red > 0, "Should have some red")
                #expect(abs(rgb.blue) <= 1e-6, "Blue should be 0 at ≤1900K")
        }
}
