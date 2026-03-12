///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
public protocol FramedBsdf: BsdfFrameProtocol, DistributionModel, Bsdf, Sendable {
        func evaluateWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum
        func probabilityDensityWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> Real
        func sampleWorldSpace(outgoing outgoingWorld: Vector, uSample: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
}

extension FramedBsdf {

        public func evaluateWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum {
                var totalLightScattered = black
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                let incidentLocal = worldToLocal(world: incidentWorld)
                let reflect = isReflecting(incident: incidentWorld, outgoing: outgoingWorld)
                if reflect && isReflective {
                        totalLightScattered += evaluate(outgoing: outgoingLocal, incident: incidentLocal)
                }
                if !reflect && isTransmissive {
                        totalLightScattered += evaluate(outgoing: outgoingLocal, incident: incidentLocal)
                }
                return totalLightScattered
        }

        public func probabilityDensityWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> Real {
                let incidentLocal = worldToLocal(world: incidentWorld)
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                if outgoingLocal.z == 0 { return 0 }
                return probabilityDensity(outgoing: outgoingLocal, incident: incidentLocal)
        }

        public func sampleWorldSpace(outgoing outgoingWorld: Vector, uSample: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool) {
                let outgoingLocal = worldToLocal(world: outgoingWorld)
                let bsdfSample = sample(outgoing: outgoingLocal, uSample: uSample)
                let incidentWorld = localToWorld(local: bsdfSample.incoming)
                return (
                        BsdfSample(bsdfSample.estimate, incidentWorld, bsdfSample.probabilityDensity),
                        isTransmissive
                )
        }

        public func evaluateDistributionFunction(outgoing: Vector, incident: Vector, normal: Normal) -> RgbSpectrum {
                let reflected = evaluateWorldSpace(outgoing: outgoing, incident: incident)
                let dotVal = absDot(incident, Vector(normal: normal))
                let scatter = reflected * dotVal
                return scatter
        }

        public func sampleDistributionFunction(outgoing: Vector, normal: Normal, sampler: inout Sampler)
                -> BsdfSample {
                var (bsdfSample, _) = sampleWorldSpace(outgoing: outgoing, uSample: sampler.get3D())
                bsdfSample.estimate *= absDot(bsdfSample.incoming, normal)
                return bsdfSample
        }

        public func evaluateProbabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                return probabilityDensityWorldSpace(outgoing: outgoing, incident: incident)
        }
}
