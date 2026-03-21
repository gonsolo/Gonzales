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
                var state = UInt64(bitPattern: Int64(pixel.x)) &* 0x85EBCA77C2B2AE63
                state ^= UInt64(bitPattern: Int64(pixel.y)) &* 0xC2B2AE3D27D4EB4F
                state ^= UInt64(bitPattern: Int64(index)) &* 0x27D4EB2F165667C5
                
                func splitmix64() -> UInt64 {
                        state &+= 0x9e3779b97f4a7c15
                        var z = state
                        z = (z ^ (z &>> 30)) &* 0xBF58476D1CE4E5B9
                        z = (z ^ (z &>> 27)) &* 0x94D049BB133111EB
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
        -> ZSobolSampler {
        var samples = try parameters.findOneInt(called: "pixelsamples", else: 1)
        if quick {
                samples = 1
        }
        return ZSobolSampler(samplesPerPixel: samples, fullResolution: fullResolution, seed: 1234)
}
