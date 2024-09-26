enum LightSampler {

        case power(PowerLightSampler)
        case uniform(UniformLightSampler)

        @MainActor
        func chooseLight() async -> (Light, FloatX) {
                switch self {
                case .power(let powerLightSampler):
                        return await powerLightSampler.chooseLight()
                case .uniform(let uniformLightSampler):
                        return uniformLightSampler.chooseLight()
                }
        }

}
