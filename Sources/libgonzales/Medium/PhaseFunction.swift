protocol PhaseFunction: DistributionModel, Sendable {
        func evaluate(outgoing: Vector, incident: Vector) -> FloatX
        func samplePhase(outgoing: Vector, sampler: inout Sampler) -> (value: FloatX, incident: Vector)
}

extension PhaseFunction {

        func evaluateDistributionFunction(outgoing: Vector, incident: Vector, normal _: Normal = zeroNormal)
                -> RgbSpectrum {
                let phase = evaluate(outgoing: outgoing, incident: incident)
                let scatter = RgbSpectrum(intensity: phase)
                return scatter
        }

        func sampleDistributionFunction(
                outgoing: Vector, normal _: Normal = zeroNormal, sampler: inout Sampler
        )
                -> BsdfSample {
                let (value, incident) = samplePhase(outgoing: outgoing, sampler: &sampler)
                return BsdfSample(RgbSpectrum(intensity: value), incident, value)
        }

        func evaluateProbabilityDensity(outgoing: Vector, incident: Vector) -> FloatX {
                return evaluate(outgoing: outgoing, incident: incident)
        }
}
