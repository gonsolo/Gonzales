struct LightSampler: Sendable {

        var powerLightSampler: PowerLightSampler

        mutating func chooseLight(scene: Scene) -> (Light, Real) {
                return powerLightSampler.chooseLight(scene: scene)
        }
}
