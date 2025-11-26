public enum Sampler: Sendable {
        case random(RandomSampler)
        case sobol(ZSobolSampler)

        mutating func get1D() -> RandomVariable {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get1D()
                        self = .random(randomSampler)
                        return value
                case .sobol(var zSobolSampler):
                        let value = zSobolSampler.get1D()
                        self = .sobol(zSobolSampler)
                        return value
                }
        }

        mutating func get2D() -> TwoRandomVariables {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get2D()
                        self = .random(randomSampler)
                        return value
                case .sobol(var zSobolSampler):
                        let value = zSobolSampler.get2D()
                        self = .sobol(zSobolSampler)
                        return value
                }
        }

        mutating func get3D() -> ThreeRandomVariables {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get3D()
                        self = .random(randomSampler)
                        return value
                case .sobol(var zSobolSampler):
                        let value = zSobolSampler.get3D()
                        self = .sobol(zSobolSampler)
                        return value
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

        func clone() -> Sampler {
                switch self {
                case .random(let randomSampler):
                        return .random(randomSampler.clone())
                case .sobol(let zSobolSampler):
                        return .sobol(zSobolSampler.clone())
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

        mutating func startPixelSample(pixel: Point2i, index: Int, dim: Int = 0) {
                switch self {
                case .random:
                        break
                case .sobol(var zSobolSampler):
                        zSobolSampler.startPixelSample(pixel: pixel, index: index, dim: dim)
                        self = .sobol(zSobolSampler)
                }
        }

}

public typealias RandomVariable = FloatX
public typealias TwoRandomVariables = (RandomVariable, RandomVariable)
public typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)
// swiftlint:disable:next large_tuple
public typealias FourRandomVariables = (RandomVariable, RandomVariable, RandomVariable, RandomVariable)
