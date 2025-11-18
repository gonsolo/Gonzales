struct RandomSampler {

        mutating func getCameraSample(pixel: Point2i, filter: Filter) -> CameraSample {
                let ux = get1D()
                let uy = get1D()

                let filterSample: FilterSample = filter.sample(u: (ux, uy))

                let dx = filterSample.location.x
                let dy = filterSample.location.y

                return CameraSample(
                        film: (
                                FloatX(pixel.x) + 0.5 + dx,
                                FloatX(pixel.y) + 0.5 + dy
                        ),
                        lens: get2D(),
                        filterWeight: filterSample.probabilityDensity
                )

        }

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
