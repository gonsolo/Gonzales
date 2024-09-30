protocol PhaseFunction: Sendable {
        func evaluate(wo: Vector, wi: Vector) -> FloatX
        func samplePhase(wo: Vector, sampler: Sampler) async -> (value: FloatX, wi: Vector)
}
