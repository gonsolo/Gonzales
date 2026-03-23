public enum Sampler: Sendable {
        case random(RandomSampler)
        case sobol(ZSobolSampler)

        mutating func get1D() -> RandomVariable {
                switch self {
                case .random(var value):
                        let result = value.get1D()
                        self = .random(value)
                        return result
                case .sobol(var value):
                        let result = value.get1D()
                        self = .sobol(value)
                        return result
                }
        }

        mutating func get2D() -> TwoRandomVariables {
                switch self {
                case .random(var value):
                        let result = value.get2D()
                        self = .random(value)
                        return result
                case .sobol(var value):
                        let result = value.get2D()
                        self = .sobol(value)
                        return result
                }
        }

        mutating func get3D() -> ThreeRandomVariables {
                switch self {
                case .random(var value):
                        let result = value.get3D()
                        self = .random(value)
                        return result
                case .sobol(var value):
                        let result = value.get3D()
                        self = .sobol(value)
                        return result
                }
        }

        var samplesPerPixel: Int {
                switch self {
                case .random(let value): return value.samplesPerPixel
                case .sobol(let value): return 1 << value.log2SamplesPerPixel
                }
        }

        func clone() -> Sampler {
                switch self {
                case .random(let value): return .random(value.clone())
                case .sobol(let value): return .sobol(value.clone())
                }
        }

        mutating func startPixelSample(pixel: Point2i, index: Int, dim: Int = 0) {
                switch self {
                case .random(var value):
                        value.startPixelSample(pixel: pixel, index: index)
                        self = .random(value)
                case .sobol(var value):
                        value.startPixelSample(pixel: pixel, index: index, dim: dim)
                        self = .sobol(value)
                }
        }

        mutating func getCameraSample(pixel: Point2i, filter: Filter) -> CameraSample {
                let (sampleX, sampleY) = get2D()
                let filterSample: FilterSample = filter.sample(uSample: (sampleX, sampleY))
                let deltaX = filterSample.location.x
                let deltaY = filterSample.location.y
                return CameraSample(
                        film: (
                                Real(pixel.x) + 0.5 + deltaX,
                                Real(pixel.y) + 0.5 + deltaY
                        ),
                        lens: get2D(),
                        filterWeight: filterSample.probabilityDensity
                )
        }
}

public typealias RandomVariable = Real
public typealias TwoRandomVariables = (RandomVariable, RandomVariable)
public typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)
// swiftlint:disable:next large_tuple
public typealias FourRandomVariables = (RandomVariable, RandomVariable, RandomVariable, RandomVariable)
