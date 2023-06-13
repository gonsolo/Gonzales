struct MediumInteraction: Interaction {

        func evaluateDistributionFunction(wi: Vector) -> RGBSpectrum {
                let phase = phase.evaluate(wo: wo, wi: wi)
                let scatter = RGBSpectrum(intensity: phase)
                return scatter
        }

        func sampleDistributionFunction(sampler: Sampler) -> BsdfSample {
                let (value, wi) = phase.samplePhase(wo: wo, sampler: sampler)
                return BsdfSample(RGBSpectrum(intensity: value), wi, value)
        }

        func evaluateProbabilityDensity(wi: Vector) -> FloatX {
                return phase.evaluate(wo: wo, wi: wi)
        }

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2F()
        var wo = Vector()

        var phase: PhaseFunction = HenyeyGreenstein()
}
