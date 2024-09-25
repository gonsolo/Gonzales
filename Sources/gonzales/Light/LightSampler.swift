//protocol LightSampler {
//        func chooseLight() -> (Light, FloatX)
//}

enum LightSampler {

        case power(PowerLightSampler)
        case uniform(UniformLightSampler)

        @MainActor
        func chooseLight() -> (Light, FloatX) {
                switch self {
                case .power(let powerLightSampler):
                        return powerLightSampler.chooseLight()
                case .uniform(let uniformLightSampler):
                        return uniformLightSampler.chooseLight()
                }
        }

}
