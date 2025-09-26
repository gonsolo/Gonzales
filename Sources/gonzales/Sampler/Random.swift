//struct RandomSampler: Sampler {
struct RandomSampler {
        func getCameraSample(pixel: Point2i) -> CameraSample {
                return CameraSample(
                        film: (
                                FloatX(pixel.x) + get1D(),
                                FloatX(pixel.y) + get1D()
                        ),
                        lens: get2D())
        }

        init(numberOfSamples: Int = 1) {
                samplesPerPixel = numberOfSamples
        }

        init(instance: RandomSampler) {
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

        func clone() -> RandomSampler {
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
