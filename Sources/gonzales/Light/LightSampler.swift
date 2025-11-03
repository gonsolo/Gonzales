enum LightSampler: Sendable {

        case power(PowerLightSampler)
        case uniform(UniformLightSampler)

        func chooseLight(scene: Scene) -> (Light, FloatX) {
                switch self {
                case .power(let powerLightSampler):
                        return powerLightSampler.chooseLight(scene: scene)
                case .uniform(let uniformLightSampler):
                        return uniformLightSampler.chooseLight()
                }
        }
}
