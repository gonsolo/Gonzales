///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
public protocol GlobalBsdf: BsdfFrameProtocol, DistributionModel, LocalBsdf, Sendable {
        func evaluateWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum
        func probabilityDensityWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> FloatX
        func sampleWorld(outgoing outgoingWorld: Vector, uSample: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
}

extension GlobalBsdf {

        public func evaluateWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum {
                var totalLightScattered = black
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                let incidentLocal = worldToLocal(world: incidentWorld)
                let reflect = isReflecting(incident: incidentWorld, outgoing: outgoingWorld)
                if reflect && isReflective {
                        totalLightScattered += evaluateLocal(outgoing: outgoingLocal, incident: incidentLocal)
                }
                if !reflect && isTransmissive {
                        totalLightScattered += evaluateLocal(outgoing: outgoingLocal, incident: incidentLocal)
                }
                return totalLightScattered
        }

        public func probabilityDensityWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> FloatX {
                let incidentLocal = worldToLocal(world: incidentWorld)
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                if outgoingLocal.z == 0 { return 0 }
                return probabilityDensityLocal(outgoing: outgoingLocal, incident: incidentLocal)
        }

        public func sampleWorld(outgoing outgoingWorld: Vector, uSample: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
        {
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                let bsdfSample = sampleLocal(outgoing: outgoingLocal, uSample: uSample)
                let incidentWorld = localToWorld(local: bsdfSample.incoming)
                return (
                        BsdfSample(bsdfSample.estimate, incidentWorld, bsdfSample.probabilityDensity),
                        isTransmissive
                )
        }

        public func evaluateDistributionFunction(outgoing: Vector, incident: Vector, normal: Normal) -> RgbSpectrum {
                let reflected = evaluateWorld(outgoing: outgoing, incident: incident)
                let dotVal = absDot(incident, Vector(normal: normal))
                let scatter = reflected * dotVal
                return scatter
        }

        public func sampleDistributionFunction(outgoing: Vector, normal: Normal, sampler: inout Sampler)
                -> BsdfSample
        {
                var (bsdfSample, _) = sampleWorld(outgoing: outgoing, uSample: sampler.get3D())
                bsdfSample.estimate *= absDot(bsdfSample.incoming, normal)
                return bsdfSample
        }

        public func evaluateProbabilityDensity(outgoing: Vector, incident: Vector) -> FloatX {
                return probabilityDensityWorld(outgoing: outgoing, incident: incident)
        }
}
