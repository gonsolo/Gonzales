final class RandomSampler: Sampler {

        init(numberOfSamples: Int = 1) {
                samplesPerPixel = numberOfSamples
        }

        init(instance: Sampler) {
                samplesPerPixel = instance.samplesPerPixel
        }

        func get1D() -> RandomVariable {
                return FloatX.random(in: 0..<1, using: &xoshiro)
        }

        func get2D() -> TwoRandomVariables {
                return (get1D(), get1D())
        }

        func get3D() -> ThreeRandomVariables {
                return (get1D(), get1D(), get1D())
        }

        func clone() -> Sampler {
                return RandomSampler(numberOfSamples: samplesPerPixel)
        }

        var samplesPerPixel: Int
        var xoshiro = Xoshiro()
}

func createRandomSampler(parameters: ParameterDictionary) throws -> RandomSampler {
        let samples = try parameters.findOneInt(called: "pixelsamples", else: 1)
        return RandomSampler(numberOfSamples: samples)
}
