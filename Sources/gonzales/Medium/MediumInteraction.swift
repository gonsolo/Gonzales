struct MediumInteraction {

        func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum {
                let phase = phase.evaluate(wo: wo, wi: wi)
                let scatter = RgbSpectrum(intensity: phase)
                return scatter
        }

        func sampleDistributionFunction(sampler: RandomSampler) -> BsdfSample {
                let (value, wi) = phase.samplePhase(wo: wo, sampler: sampler)
                return BsdfSample(RgbSpectrum(intensity: value), wi, value)
        }

        func evaluateProbabilityDensity(wi: Vector) -> FloatX {
                return phase.evaluate(wo: wo, wi: wi)
        }

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2f()
        var wo = Vector()

        var phase: any PhaseFunction = HenyeyGreenstein()
}
