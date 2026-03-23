public enum Sampler: Sendable {
        case random(RandomSampler)
        case sobol(ZSobolSampler)
}

public typealias RandomVariable = Real
public typealias TwoRandomVariables = (RandomVariable, RandomVariable)
public typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)
// swiftlint:disable:next large_tuple
public typealias FourRandomVariables = (RandomVariable, RandomVariable, RandomVariable, RandomVariable)
