enum Sampler {
        case random(RandomSampler)

        mutating func get1D() -> RandomVariable  {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get1D()
                        self = .random(randomSampler)
                        return value
                }
        }

        mutating func get2D() -> TwoRandomVariables {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get2D()
                        self = .random(randomSampler)
                        return value
                }
        }

        mutating func get3D() -> ThreeRandomVariables {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.get3D()
                        self = .random(randomSampler)
                        return value
                }
        }

        var samplesPerPixel: Int {
                switch self {
                case .random(let randomSampler):
                        return randomSampler.samplesPerPixel
                }
        }

        mutating func getCameraSample(pixel: Point2i, filter: Filter) -> CameraSample  {
                switch self {
                case .random(var randomSampler):
                        let value = randomSampler.getCameraSample(pixel: pixel, filter: filter)
                        self = .random(randomSampler)
                        return value
                }
        }

        func clone() -> Sampler  {
                switch self {
                case .random(let randomSampler):
                        return .random(randomSampler.clone())
                }
        }
}

typealias RandomVariable = FloatX
typealias TwoRandomVariables = (RandomVariable, RandomVariable)
typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)
// swiftlint:disable:next large_tuple
typealias FourRandomVariables = (RandomVariable, RandomVariable, RandomVariable, RandomVariable)
