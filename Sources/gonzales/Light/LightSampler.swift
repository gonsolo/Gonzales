struct LightSampler: Sendable {

        //case power(PowerLightSampler)
        //case uniform(UniformLightSampler)

        var powerLightSampler: PowerLightSampler

        mutating func chooseLight(scene: Scene) -> (Light, FloatX) {
                return powerLightSampler.chooseLight(scene: scene)
        }

        //func chooseLight(scene: Scene) -> (Light, FloatX) {
        //        switch self {
        //        case .power(var powerLightSampler):
        //                return powerLightSampler.chooseLight(scene: scene)
        //        case .uniform(var uniformLightSampler):
        //                return uniformLightSampler.chooseLight()
        //        }
        //}
}
