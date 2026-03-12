struct LightSampler: Sendable {

        var powerLightSampler: PowerLightSampler

        mutating func chooseLight(scene: Scene) throws -> (Light, Real) {
                return try powerLightSampler.chooseLight(scene: scene)
        }
}
