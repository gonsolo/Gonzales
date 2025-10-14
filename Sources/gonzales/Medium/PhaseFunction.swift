protocol PhaseFunction: DistributionModel, Sendable {
        func evaluate(wo: Vector, wi: Vector) -> FloatX
        func samplePhase(wo: Vector, sampler: RandomSampler) -> (value: FloatX, wi: Vector)
}

extension PhaseFunction {
        //func evaluateDistributionFunction(wo: Vector, wi: Vector, normal: Normal) -> RgbSpectrum
        //func sampleDistributionFunction(wo: Vector, normal: Normal, sampler: RandomSampler) -> BsdfSample
        //func evaluateProbabilityDensity(wo: Vector, wi: Vector) -> FloatX

        func evaluateDistributionFunction(wo: Vector, wi: Vector, normal: Normal = zeroNormal) -> RgbSpectrum {
                let phase = evaluate(wo: wo, wi: wi)
                let scatter = RgbSpectrum(intensity: phase)
                return scatter
        }

        func sampleDistributionFunction(wo: Vector, normal: Normal = zeroNormal, sampler: RandomSampler) -> BsdfSample {
                let (value, wi) = samplePhase(wo: wo, sampler: sampler)
                return BsdfSample(RgbSpectrum(intensity: value), wi, value)
        }

        func evaluateProbabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                return evaluate(wo: wo, wi: wi)
        }
}

