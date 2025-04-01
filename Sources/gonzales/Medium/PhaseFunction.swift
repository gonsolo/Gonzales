protocol PhaseFunction: Sendable {
        func evaluate(wo: Vector, wi: Vector) -> FloatX
        func samplePhase(wo: Vector, sampler: any Sampler) -> (value: FloatX, wi: Vector)
}
