struct LightSampler: Sendable {

        var powerLightSampler: PowerLightSampler

        mutating func chooseLight(scene: Scene) throws -> (Light, FloatX) {
                return try powerLightSampler.chooseLight(scene: scene)
        }
}
