enum Sampler {
        case random(RandomSampler)
        case sobol(ZSobolSampler)

        func get1D() ->RandomVariable {
                switch self {
                case .random(var randomSampler):
                        return randomSampler.get1D()
                case .sobol(var zSobolSampler):
                        return zSobolSampler.get1D()
                }
        }

        func get2D() -> TwoRandomVariables {
                switch self {
                case .random(var randomSampler):
                        return randomSampler.get2D()
                case .sobol(var zSobolSampler):
                        return zSobolSampler.get2D()
                }
        }

        func get3D() -> ThreeRandomVariables {
                switch self {
                case .random(var randomSampler):
                        return randomSampler.get3D()
                case .sobol(var zSobolSampler):
                        return zSobolSampler.get3D()
                }
        }

        var samplesPerPixel: Int {
                switch self {
                case .random(let randomSampler):
                        return randomSampler.samplesPerPixel
                case .sobol(let zSobolSampler):
                        return 1 << zSobolSampler.log2SamplesPerPixel
                }
        }

        mutating func getCameraSample(pixel: Point2i, filter: Filter) -> CameraSample {
                let (ux, uy) = get2D()

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

}

typealias RandomVariable = FloatX
typealias TwoRandomVariables = (RandomVariable, RandomVariable)
typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)
// swiftlint:disable:next large_tuple
typealias FourRandomVariables = (RandomVariable, RandomVariable, RandomVariable, RandomVariable)
