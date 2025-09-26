protocol PhaseFunction: Sendable {
        func evaluate(wo: Vector, wi: Vector) -> FloatX
        func samplePhase(wo: Vector, sampler: RandomSampler) -> (value: FloatX, wi: Vector)
}
