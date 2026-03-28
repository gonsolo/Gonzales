public struct RandomSampler: Sendable {

        init(numberOfSamples: Int = 1) {
                samplesPerPixel = numberOfSamples
                xoshiro = Xoshiro()
        }

        init(instance: RandomSampler) {
                samplesPerPixel = instance.samplesPerPixel
                xoshiro = Xoshiro()
        }

        mutating func get1D() -> RandomVariable {
                return Real.random(in: 0..<1, using: &xoshiro)
        }

        mutating func get2D() -> TwoRandomVariables {
                return (get1D(), get1D())
        }

        mutating func get3D() -> ThreeRandomVariables {
                return (get1D(), get1D(), get1D())
        }

        mutating func startPixelSample(pixel: Point2i, index: Int) {
                var state = UInt64(bitPattern: Int64(pixel.x)) &* 0x85EB_CA77_C2B2_AE63
                state ^= UInt64(bitPattern: Int64(pixel.y)) &* 0xC2B2_AE3D_27D4_EB4F
                state ^= UInt64(bitPattern: Int64(index)) &* 0x27D4_EB2F_1656_67C5

                func splitmix64() -> UInt64 {
                        state &+= 0x9e37_79b9_7f4a_7c15
                        var z = state
                        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
                        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
                        return z ^ (z &>> 31)
                }

                xoshiro = Xoshiro(seed: [splitmix64(), splitmix64(), splitmix64(), splitmix64()])
        }

        func clone() -> RandomSampler {
                return RandomSampler(numberOfSamples: samplesPerPixel)
        }

        let samplesPerPixel: Int
        var xoshiro: Xoshiro
}

func createRandomSampler(parameters: ParameterDictionary, quick: Bool) throws -> RandomSampler {
        var samples = try parameters.findOneInt(called: "pixelsamples", else: 1)
        if quick {
                samples = 1
        }
        return RandomSampler(numberOfSamples: samples)
}

func createZSobolSampler(parameters: ParameterDictionary, fullResolution: Point2i, quick: Bool) throws
        -> ZSobolSampler
{
        var samples = try parameters.findOneInt(called: "pixelsamples", else: 1)
        if quick {
                samples = 1
        }
        return ZSobolSampler(samplesPerPixel: samples, fullResolution: fullResolution, seed: 1234)
}
