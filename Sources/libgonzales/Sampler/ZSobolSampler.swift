import Foundation  // For Point2i, etc.

let floatOneMinusEpsilon: Float = 1.0 - 1e-6  // Ensure type is Float

func sobolSample(sampleIndex: Int, dimension: Int, randomizer: FastOwenScrambler) -> Float {
        guard dimension < NSobolDimensions else {
                fatalError("Sobol dimension \(dimension) is out of range (\(NSobolDimensions)).")
        }

        var accumulator: UInt32 = 0
        var currentIndex = sampleIndex

        let matrixAccessor = sobolDataAccessor[dimension]

        for bitIndex in 0..<SobolMatrixSize {
                if currentIndex & 1 != 0 {
                        // Direct access: no temporary arrays, just fast index math
                        accumulator ^= matrixAccessor[bitIndex]
                }
                currentIndex >>= 1
                if currentIndex == 0 {
                        break
                }
        }

        let randomizedV = randomizer(accumulator)
        let floatValue = Float(randomizedV) * Float(2.32830643653869628906e-10)

        return min(floatValue, floatOneMinusEpsilon)
}

func reverseBits32(_ valueIn: UInt32) -> UInt32 {
        var value = valueIn
        value = ((value >> 1) & 0x5555_5555) | ((value & 0x5555_5555) << 1)
        value = ((value >> 2) & 0x3333_3333) | ((value & 0x3333_3333) << 2)
        value = ((value >> 4) & 0x0f0f_0f0f) | ((value & 0x0f0f_0f0f) << 4)
        value = ((value >> 8) & 0x00ff_00ff) | ((value & 0x00ff_00ff) << 8)
        return (value >> 16) | (value << 16)
}

struct FastOwenScrambler {
        let seed: UInt32

        /**
         * Initializes the scrambler with a 32-bit seed.
         */
        @inlinable
        init(seed: UInt32) {
                self.seed = seed
        }

        @inlinable
        func callAsFunction(_ valueIn: UInt32) -> UInt32 {
                var value = valueIn
                value = reverseBits32(value)
                value ^= value &* 0x3d20_adea  // Use &* for checked multiplication
                value &+= seed  // Use &+ for checked addition
                let multiplier = (seed >> 16) | 1
                value &*= multiplier
                value ^= value &* 0x0552_6c56
                value ^= value &* 0x53a2_2864
                return reverseBits32(value)
        }
}

@inlinable
func encodeMorton2(_ coordX: UInt32, _ coordY: UInt32) -> UInt64 {
        var x64: UInt64 = UInt64(coordX)
        var y64: UInt64 = UInt64(coordY)

        x64 = (x64 | (x64 << 16)) & 0x0000_FFFF_0000_FFFF
        x64 = (x64 | (x64 << 8)) & 0x00FF_00FF_00FF_00FF
        x64 = (x64 | (x64 << 4)) & 0x0F0F_0F0F_0F0F_0F0F
        x64 = (x64 | (x64 << 2)) & 0x3333_3333_3333_3333
        x64 = (x64 | (x64 << 1)) & 0x5555_5555_5555_5555

        y64 = (y64 | (y64 << 16)) & 0x0000_FFFF_0000_FFFF
        y64 = (y64 | (y64 << 8)) & 0x00FF_00FF_00FF_00FF
        y64 = (y64 | (y64 << 4)) & 0x0F0F_0F0F_0F0F_0F0F
        y64 = (y64 | (y64 << 2)) & 0x3333_3333_3333_3333
        y64 = (y64 | (y64 << 1)) & 0x5555_5555_5555_5555

        return x64 | (y64 << 1)
}

@inlinable
func mixBits(_ valueIn: UInt64) -> UInt32 {
        // This is a minimal, quick hash based on PBRT's use of MixBits
        var value32 = UInt32(valueIn & 0xFFFF_FFFF)
        value32 ^= UInt32(valueIn >> 32)
        value32 ^= value32 >> 16
        value32 &*= 0x85eb_ca77  // Use &* for wraparound multiplication
        value32 ^= value32 >> 13
        value32 &*= 0xc2b2_ae35
        value32 ^= value32 >> 16
        return value32
}

public struct ZSobolSampler: Sendable {

        private static let permutations: [[UInt8]] = [
                [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
                [0, 3, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3], [1, 0, 3, 2],
                [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 2, 0], [1, 3, 0, 2],
                [2, 1, 0, 3], [2, 1, 3, 0], [2, 0, 1, 3], [2, 0, 3, 1],
                [2, 3, 0, 1], [2, 3, 1, 0], [3, 1, 2, 0], [3, 1, 0, 2],
                [3, 2, 1, 0], [3, 2, 0, 1], [3, 0, 2, 1], [3, 0, 1, 2],
        ]

        init(samplesPerPixel: Int, fullResolution: Point2i, seed: Int) {
                self.log2SamplesPerPixel = max(0, samplesPerPixel.trailingZeroBitCount)
                self.seed = seed

                let res = max(fullResolution.x, fullResolution.y).nextPowerOfTwo()
                let log4SamplesPerPixel = (self.log2SamplesPerPixel + 1) / 2

                self.nBase4Digits = res.trailingZeroBitCount + log4SamplesPerPixel
        }

        init(seed: Int, log2SamplesPerPixel: Int, nBase4Digits: Int) {
                self.seed = seed
                self.log2SamplesPerPixel = log2SamplesPerPixel
                self.nBase4Digits = nBase4Digits
        }

        mutating func startPixelSample(pixel: Point2i, index: Int, dim: Int) {
                self.dimension = dim
                let pixelX = UInt32(pixel.x)
                let pixelY = UInt32(pixel.y)
                let pixelMortonIndex = encodeMorton2(pixelX, pixelY)
                self.mortonIndex = (pixelMortonIndex << log2SamplesPerPixel) | UInt64(index)
        }

        func getSampleIndex() -> UInt64 {
                var sampleIndex: UInt64 = 0
                let pow2Samples = log2SamplesPerPixel & 1 == 1
                let lastDigit = pow2Samples ? 1 : 0

                // Loop over base-4 digits for permutation
                for digitIndex in (lastDigit..<nBase4Digits).reversed() {
                        let digitShift = 2 * digitIndex - (pow2Samples ? 1 : 0)
                        var digit = Int((mortonIndex >> digitShift) & 3)
                        let higherDigits = mortonIndex >> (digitShift + 2)

                        // --- Direct Owen/Permutation Logic (MixBits) ---
                        let hashVal = mixBits(higherDigits ^ (UInt64(0x5555_5555) * UInt64(dimension)))
                        let pIndex = Int((hashVal >> 24) % 24)

                        digit = Int(ZSobolSampler.permutations[pIndex][digit])
                        sampleIndex |= UInt64(digit) << digitShift
                }

                // --- Direct Fast Owen Logic for odd log2SamplesPerPixel ---
                if pow2Samples {
                        var digit = Int(mortonIndex & 1)
                        let hashVal = mixBits((mortonIndex >> 1) ^ (UInt64(0x5555_5555) * UInt64(dimension)))
                        digit ^= Int(hashVal & 1)
                        sampleIndex |= UInt64(digit)
                }

                return sampleIndex
        }

        func clone() -> ZSobolSampler {
                return ZSobolSampler(
                        seed: self.seed, log2SamplesPerPixel: self.log2SamplesPerPixel,
                        nBase4Digits: self.nBase4Digits)
        }

        let seed: Int
        let log2SamplesPerPixel: Int
        let nBase4Digits: Int

        var mortonIndex: UInt64 = 0
        var dimension: Int = 0
}

extension ZSobolSampler {

        // MARK: - Refactored Common Logic

        /// Generates the two 32-bit seeds (hashes) required for Fast Owen Scrambling
        /// for a given pair of dimensions.
        private func getSampleHash(dimensionOffset: Int) -> (UInt32, UInt32) {
                // PBRT uses the current dimension index (dimension + offset)
                let dim = self.dimension + dimensionOffset

                // Generate a 64-bit hash based on the current dimension and the sampler's seed
                let bits = UInt64(mixBits(UInt64(dim) ^ UInt64(self.seed)))

                let sampleHash0 = UInt32(bits & 0xFFFF_FFFF)
                let sampleHash1 = UInt32(bits >> 32)

                return (sampleHash0, sampleHash1)
        }

        // MARK: - Sampler Methods

        mutating func get1D() -> RandomVariable {
                let sampleIndex = Int(getSampleIndex())

                let (sampleHash, _) = getSampleHash(dimensionOffset: 0)  // Only need the first 32 bits

                if dimension >= NSobolDimensions {
                        dimension = 2
                }
                dimension += 1

                let uSample = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 1,  // Pass the correct dimension index
                        randomizer: FastOwenScrambler(seed: sampleHash)
                )
                return uSample
        }

        mutating func get2D() -> (Float, Float) {
                let sampleIndex = Int(getSampleIndex())

                let (sampleHash0, sampleHash1) = getSampleHash(dimensionOffset: 0)

                if dimension + 1 >= NSobolDimensions {
                        dimension = 2
                }
                dimension += 2

                let sample0 = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 2,  // Dimension 'd'
                        randomizer: FastOwenScrambler(seed: sampleHash0)
                )

                let sample1 = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 1,  // Dimension 'd + 1'
                        randomizer: FastOwenScrambler(seed: sampleHash1)
                )

                return (sample0, sample1)
        }
}

extension ZSobolSampler {

        mutating func get3D() -> ThreeRandomVariables {
                let sampleIndex = Int(getSampleIndex())

                let (sampleHash0, sampleHash1) = getSampleHash(dimensionOffset: 0)

                let (sampleHash2, _) = getSampleHash(dimensionOffset: 2)

                if dimension + 2 >= NSobolDimensions {
                        dimension = 2
                }
                dimension += 3

                let sample0 = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 3,
                        randomizer: FastOwenScrambler(seed: sampleHash0)
                )

                let sample1 = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 2,
                        randomizer: FastOwenScrambler(seed: sampleHash1)
                )

                let sample2 = sobolSample(
                        sampleIndex: sampleIndex,
                        dimension: dimension - 1,
                        randomizer: FastOwenScrambler(seed: sampleHash2)
                )

                return (sample0, sample1, sample2)
        }
}

extension Int {
        func nextPowerOfTwo() -> Int {
                var value = self
                value -= 1
                value |= value >> 1
                value |= value >> 2
                value |= value >> 4
                value |= value >> 8
                value |= value >> 16
                value += 1
                return value
        }
}
