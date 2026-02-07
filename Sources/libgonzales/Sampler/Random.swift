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
                return FloatX.random(in: 0..<1, using: &xoshiro)
        }

        mutating func get2D() -> TwoRandomVariables {
                return (get1D(), get1D())
        }

        mutating func get3D() -> ThreeRandomVariables {
                return (get1D(), get1D(), get1D())
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
