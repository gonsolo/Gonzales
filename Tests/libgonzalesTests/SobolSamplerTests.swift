import Testing

@testable import libgonzales

@Suite struct SobolUtilityTests {

        // MARK: - reverseBits32

        @Test func reverseBits32Zero() {
                #expect(reverseBits32(0) == 0)
        }

        @Test func reverseBits32AllOnes() {
                #expect(reverseBits32(0xFFFF_FFFF) == 0xFFFF_FFFF)
        }

        @Test func reverseBits32HighBitBecomesLowBit() {
                #expect(reverseBits32(0x8000_0000) == 1)
        }

        @Test func reverseBits32Involution() {
                let values: [UInt32] = [0, 1, 42, 0xDEAD_BEEF, 0x1234_5678, 0xFFFF_FFFF]
                for v in values {
                        #expect(
                                reverseBits32(reverseBits32(v)) == v,
                                "Double reverse should be identity for \(v)")
                }
        }

        // MARK: - encodeMorton2

        @Test func encodeMorton2Origin() {
                #expect(encodeMorton2(0, 0) == 0)
        }

        @Test func encodeMorton2InterleavesCorrectly() {
                // (1,0) -> bit 0 set -> 1
                #expect(encodeMorton2(1, 0) == 1)
                // (0,1) -> bit 1 set -> 2
                #expect(encodeMorton2(0, 1) == 2)
                // (1,1) -> bits 0+1 set -> 3
                #expect(encodeMorton2(1, 1) == 3)
        }

        // MARK: - mixBits

        @Test func mixBitsConsistentForSameInput() {
                let a = mixBits(12345)
                let b = mixBits(12345)
                #expect(a == b)
        }

        @Test func mixBitsDifferentForDifferentInputs() {
                let a = mixBits(0)
                let b = mixBits(1)
                let c = mixBits(99999)
                #expect(a != b, "Hash of 0 and 1 should differ")
                #expect(a != c, "Hash of 0 and 99999 should differ")
                #expect(b != c, "Hash of 1 and 99999 should differ")
        }

        // MARK: - nextPowerOfTwo

        @Test func nextPowerOfTwoAlreadyPower() {
                #expect(16.nextPowerOfTwo() == 16)
                #expect(1.nextPowerOfTwo() == 1)
                #expect(1024.nextPowerOfTwo() == 1024)
        }

        @Test func nextPowerOfTwoNonPower() {
                #expect(3.nextPowerOfTwo() == 4)
                #expect(5.nextPowerOfTwo() == 8)
                #expect(17.nextPowerOfTwo() == 32)
                #expect(100.nextPowerOfTwo() == 128)
        }

        // MARK: - FastOwenScrambler

        @Test func fastOwenScramblerDeterministic() {
                let scrambler = FastOwenScrambler(seed: 42)
                let a = scrambler(12345)
                let b = scrambler(12345)
                #expect(a == b)
        }

        @Test func fastOwenScramblerDifferentSeeds() {
                let scrambler1 = FastOwenScrambler(seed: 1)
                let scrambler2 = FastOwenScrambler(seed: 2)
                let result1 = scrambler1(12345)
                let result2 = scrambler2(12345)
                #expect(result1 != result2, "Different seeds should produce different outputs")
        }

        // MARK: - ZSobolSampler

        @Test func sobolSamplerValuesInRange() {
                var sampler = ZSobolSampler(
                        samplesPerPixel: 4,
                        fullResolution: Point2i(x: 64, y: 64),
                        seed: 0)
                sampler.startPixelSample(
                        pixel: Point2i(x: 10, y: 10),
                        index: 0, dim: 0)
                for _ in 0..<20 {
                        let value = sampler.get1D()
                        #expect(value >= 0, "Sobol sample below 0")
                        #expect(value < 1, "Sobol sample >= 1")
                }
        }

        @Test func sobolSampler2DValuesInRange() {
                var sampler = ZSobolSampler(
                        samplesPerPixel: 4,
                        fullResolution: Point2i(x: 64, y: 64),
                        seed: 0)
                sampler.startPixelSample(
                        pixel: Point2i(x: 5, y: 5),
                        index: 0, dim: 0)
                for _ in 0..<10 {
                        let (u, v) = sampler.get2D()
                        #expect(u >= 0 && u < 1, "Sobol 2D u out of range")
                        #expect(v >= 0 && v < 1, "Sobol 2D v out of range")
                }
        }

        @Test func sobolSampler3DValuesInRange() {
                var sampler = ZSobolSampler(
                        samplesPerPixel: 4,
                        fullResolution: Point2i(x: 64, y: 64),
                        seed: 0)
                sampler.startPixelSample(
                        pixel: Point2i(x: 5, y: 5),
                        index: 0, dim: 0)
                for _ in 0..<10 {
                        let (u, v, w) = sampler.get3D()
                        #expect(u >= 0 && u < 1, "Sobol 3D u out of range")
                        #expect(v >= 0 && v < 1, "Sobol 3D v out of range")
                        #expect(w >= 0 && w < 1, "Sobol 3D w out of range")
                }
        }
}
