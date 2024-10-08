enum LightSampler: Sendable {

        case power(PowerLightSampler)
        case uniform(UniformLightSampler)

        func chooseLight() -> (Light, FloatX) {
                switch self {
                case .power(let powerLightSampler):
                        return powerLightSampler.chooseLight()
                case .uniform(let uniformLightSampler):
                        return uniformLightSampler.chooseLight()
                }
        }

}
