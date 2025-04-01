struct RandomSampler: Sampler {

        init(numberOfSamples: Int = 1) {
                samplesPerPixel = numberOfSamples
        }

        init(instance: any Sampler) {
                samplesPerPixel = instance.samplesPerPixel
        }

        func get1D() -> RandomVariable {
                //return FloatX.random(in: 0..<1, using: &xoshiro)
                return FloatX.random(in: 0..<1)
        }

        func get2D() -> TwoRandomVariables {
                return (get1D(), get1D())
        }

        func get3D() -> ThreeRandomVariables {
                return (get1D(), get1D(), get1D())
        }

        func clone() -> any Sampler {
                return RandomSampler(numberOfSamples: samplesPerPixel)
        }

        let samplesPerPixel: Int
        //var xoshiro = Xoshiro()
}

func createRandomSampler(parameters: ParameterDictionary, quick: Bool) throws -> RandomSampler {
        var samples = try parameters.findOneInt(called: "pixelsamples", else: 1)
        if quick {
                samples = 1
        }
        return RandomSampler(numberOfSamples: samples)
}
