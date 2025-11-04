enum LightSampler: Sendable {

        case power(PowerLightSampler)
        case uniform(UniformLightSampler)

        func chooseLight(scene: Scene) -> (Light, FloatX) {
                switch self {
                case .power(var powerLightSampler):
                        return powerLightSampler.chooseLight(scene: scene)
                case .uniform(var uniformLightSampler):
                        return uniformLightSampler.chooseLight()
                }
        }
}
