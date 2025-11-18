let SobolMatrixSize = 52
let NSobolDimensions = 1024
let FloatOneMinusEpsilon: Float = 1.0 - 1e-6 // Ensure type is Float

func sobolSample(a: Int, dimension: Int, randomizer: FastOwenScrambler) -> Float 
{
    guard dimension < NSobolDimensions else {
        fatalError("Sobol dimension \(dimension) is out of range (\(NSobolDimensions)).")
    }

    var v: UInt32 = 0
    var index = a
    
    let matrixStartIndex = dimension * SobolMatrixSize
    
    for i in matrixStartIndex..<(matrixStartIndex + SobolMatrixSize) {
        if index & 1 != 0 {
            v ^= sobolMatrices[i]
        }
        index >>= 1
        if index == 0 {
            break
        }
    }
    
    let randomizedV = randomizer(v)
    let floatValue = Float(randomizedV) * Float(2.32830643653869628906e-10) 
    
    return min(floatValue, FloatOneMinusEpsilon)
}

func reverseBits32(_ v: UInt32) -> UInt32 {
    var v = v
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1) 
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2) 
    v = ((v >> 4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4) 
    v = ((v >> 8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8) 
    return (v >> 16) | (v << 16) 
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
    func callAsFunction(_ vIn: UInt32) -> UInt32 {
        var v = vIn
        v = reverseBits32(v)
        v ^= v &* 0x3d20adea // Use &* for checked multiplication
        v &+= seed // Use &+ for checked addition
        let multiplier = (seed >> 16) | 1
        v &*= multiplier
        v ^= v &* 0x05526c56
        v ^= v &* 0x53a22864
        return reverseBits32(v)
    }
}

@inlinable
func encodeMorton2(_ x: UInt32, _ y: UInt32) -> UInt64 {
    var x64: UInt64 = UInt64(x)
    var y64: UInt64 = UInt64(y)

    x64 = (x64 | (x64 << 16)) & 0x0000FFFF0000FFFF
    x64 = (x64 | (x64 << 8))  & 0x00FF00FF00FF00FF
    x64 = (x64 | (x64 << 4))  & 0x0F0F0F0F0F0F0F0F
    x64 = (x64 | (x64 << 2))  & 0x3333333333333333
    x64 = (x64 | (x64 << 1))  & 0x5555555555555555

    y64 = (y64 | (y64 << 16)) & 0x0000FFFF0000FFFF
    y64 = (y64 | (y64 << 8))  & 0x00FF00FF00FF00FF
    y64 = (y64 | (y64 << 4))  & 0x0F0F0F0F0F0F0F0F
    y64 = (y64 | (y64 << 2))  & 0x3333333333333333
    y64 = (y64 | (y64 << 1))  & 0x5555555555555555

    return x64 | (y64 << 1)
}

@inlinable
func mixBits(_ v: UInt64) -> UInt32 {
    // This is a minimal, quick hash based on PBRT's use of MixBits
    var v32 = UInt32(v & 0xFFFFFFFF)
    v32 ^= UInt32(v >> 32)
    v32 ^= v32 >> 16
    v32 &*= 0x85ebca77 // Use &* for wraparound multiplication
    v32 ^= v32 >> 13
    v32 &*= 0xc2b2ae35
    v32 ^= v32 >> 16
    return v32
}

import Foundation // For Point2i, etc.

struct ZSobolSampler {
    let seed: Int
    let log2SamplesPerPixel: Int
    let nBase4Digits: Int

    var mortonIndex: UInt64 = 0
    var dimension: Int = 0

    private static let permutations: [[UInt8]] = [
        [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
        [0, 3, 2, 1], [0, 3, 1, 2], [1, 0, 2, 3], [1, 0, 3, 2],
        [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 2, 0], [1, 3, 0, 2],
        [2, 1, 0, 3], [2, 1, 3, 0], [2, 0, 1, 3], [2, 0, 3, 1],
        [2, 3, 0, 1], [2, 3, 1, 0], [3, 1, 2, 0], [3, 1, 0, 2],
        [3, 2, 1, 0], [3, 2, 0, 1], [3, 0, 2, 1], [3, 0, 1, 2]
    ]

    init(samplesPerPixel: Int, fullResolution: Point2i, seed: Int) {
        self.log2SamplesPerPixel = max(0, samplesPerPixel.trailingZeroBitCount)
        self.seed = seed

        let res = max(fullResolution.x, fullResolution.y).nextPowerOfTwo()
        let log4SamplesPerPixel = (self.log2SamplesPerPixel + 1) / 2

        self.nBase4Digits = res.trailingZeroBitCount + log4SamplesPerPixel
    }

    // Equivalent to PBRT's StartPixelSample
    mutating func startPixelSample(pixel: (Int, Int), index: Int, dim: Int) {
        self.dimension = dim
        let pX = UInt32(pixel.0)
        let pY = UInt32(pixel.1)

        // mortonIndex = (EncodeMorton2(p.x, p.y) << log2SamplesPerPixel) | index;
        let pixelMortonIndex = encodeMorton2(pX, pY)
        self.mortonIndex = (pixelMortonIndex << log2SamplesPerPixel) | UInt64(index)
    }

    // Equivalent to PBRT's GetSampleIndex()
    // This function inherently implements the Owen-based Z-order permutation logic.
    @inlinable
    func getSampleIndex() -> UInt64 {
        var sampleIndex: UInt64 = 0
        let pow2Samples = log2SamplesPerPixel & 1 == 1
        let lastDigit = pow2Samples ? 1 : 0

        // Loop over base-4 digits for permutation
        for i in (lastDigit..<nBase4Digits).reversed() {
            let digitShift = 2 * i - (pow2Samples ? 1 : 0)
            var digit = Int((mortonIndex >> digitShift) & 3)
            let higherDigits = mortonIndex >> (digitShift + 2)

            // --- Direct Owen/Permutation Logic (MixBits) ---
            let hashVal = mixBits(higherDigits ^ (UInt64(0x55555555) * UInt64(dimension)))
            let p = Int((hashVal >> 24) % 24)

            digit = Int(ZSobolSampler.permutations[p][digit])
            sampleIndex |= UInt64(digit) << digitShift
        }

        // --- Direct Fast Owen Logic for odd log2SamplesPerPixel ---
        if pow2Samples {
            var digit = Int(mortonIndex & 1)
            let hashVal = mixBits((mortonIndex >> 1) ^ (UInt64(0x55555555) * UInt64(dimension)))
            digit ^= Int(hashVal & 1)
            sampleIndex |= UInt64(digit)
        }

        return sampleIndex
    }

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

        let sampleHash0 = UInt32(bits & 0xFFFFFFFF)
        let sampleHash1 = UInt32(bits >> 32)

        return (sampleHash0, sampleHash1)
    }

    // MARK: - Sampler Methods

    /// Generates a single-dimensional sample (Float).
    mutating func get1D() -> RandomVariable {
        // 1. Get the unified sample index
        let sampleIndex = Int(getSampleIndex())

        // 2. Generate the hash seed for the current dimension (offset 0)
        let (sampleHash, _) = getSampleHash(dimensionOffset: 0) // Only need the first 32 bits

        // 3. Increment the dimension for the next sample
        dimension += 1

        // 4. Generate the sample using the Sobol sequence and the seed
        // We use the current dimension index (dimension - 1) before incrementing
        let u = sobolSample(
            a: sampleIndex,
            dimension: dimension - 1, // Pass the correct dimension index
            randomizer: FastOwenScrambler(seed: sampleHash)
        )
        return u
    }

    /// Generates a two-dimensional sample ((Float, Float)).
    mutating func get2D() -> (Float, Float) {
        // 1. Get the unified sample index
        let sampleIndex = Int(getSampleIndex())

        // 2. Generate the hash seeds for the two dimensions (offset 0)
        let (sampleHash0, sampleHash1) = getSampleHash(dimensionOffset: 0)

        // 3. Increment the dimension by 2 for the next sample
        // Note: The sobolSample calls below use dimension and dimension + 1 before incrementing
        dimension += 2

        // 4. Generate the samples using the Sobol sequence and the seeds
        let u0 = sobolSample(
            a: sampleIndex,
            dimension: dimension - 2, // Dimension 'd'
            randomizer: FastOwenScrambler(seed: sampleHash0)
        )

        let u1 = sobolSample(
            a: sampleIndex,
            dimension: dimension - 1, // Dimension 'd + 1'
            randomizer: FastOwenScrambler(seed: sampleHash1)
        )

        return (u0, u1)
    }
}

extension ZSobolSampler {

    mutating func get3D() -> ThreeRandomVariables {
        // 1. Get the unified sample index
        let sampleIndex = Int(getSampleIndex())

        // 2. Generate the hash seeds for the three required dimensions.
        // We call getSampleHash once to get two seeds (d, d+1) and then again
        // for the third seed (d+2).

        // Hashes for dimension 'd' and 'd+1'
        let (sampleHash0, sampleHash1) = getSampleHash(dimensionOffset: 0)

        // Hash for dimension 'd+2' (We only need the first 32 bits from this call)
        let (sampleHash2, _) = getSampleHash(dimensionOffset: 2)

        // 3. Increment the dimension by 3 for the next sample
        // The samples below will use dimensions d, d+1, and d+2 (or dimension-3, dimension-2, dimension-1).
        dimension += 3

        // 4. Generate the three samples using Sobol sequences and the seeds

        // Sample for dimension 'd'
        let u0 = sobolSample(
            a: sampleIndex,
            dimension: dimension - 3,
            randomizer: FastOwenScrambler(seed: sampleHash0)
        )

        // Sample for dimension 'd+1'
        let u1 = sobolSample(
            a: sampleIndex,
            dimension: dimension - 2,
            randomizer: FastOwenScrambler(seed: sampleHash1)
        )

        // Sample for dimension 'd+2'
        let u2 = sobolSample(
            a: sampleIndex,
            dimension: dimension - 1,
            randomizer: FastOwenScrambler(seed: sampleHash2)
        )

        return (u0, u1, u2)
    }
}

extension Int {
    func nextPowerOfTwo() -> Int {
        var v = self
        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v += 1
        return v
    }
}

