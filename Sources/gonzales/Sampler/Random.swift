final class RandomSampler: Sampler {

        init(numberOfSamples: Int) {
                samplesPerPixel = numberOfSamples
        }

        init(instance: Sampler) {
                samplesPerPixel = instance.samplesPerPixel
        }

        func get1D() -> FloatX {
                return FloatX.random(in: 0..<1, using: &xoshiro)
        }

        func get2D() -> Point2F {
                return Point2F(x: get1D(), y: get1D())
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
