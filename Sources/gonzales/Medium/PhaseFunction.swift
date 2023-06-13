protocol PhaseFunction {
        func evaluate(wo: Vector, wi: Vector) -> FloatX
        func samplePhase(wo: Vector, sampler: Sampler) -> (value: FloatX, wi: Vector)
}
