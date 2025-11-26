///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
public protocol GlobalBsdf: BsdfFrameProtocol, DistributionModel, LocalBsdf, Sendable {
        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RgbSpectrum
        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX
        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
}

extension GlobalBsdf {

        public func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RgbSpectrum {
                var totalLightScattered = black
                let woLocal = worldToLocal(world: woWorld)
                print("woWorld, woLocal: ", woWorld, woLocal)
                let wiLocal = worldToLocal(world: wiWorld)
                let reflect = isReflecting(wi: wiWorld, wo: woWorld)
                if reflect && isReflective {
                        totalLightScattered += evaluateLocal(wo: woLocal, wi: wiLocal)
                }
                if !reflect && isTransmissive {
                        totalLightScattered += evaluateLocal(wo: woLocal, wi: wiLocal)
                }
                return totalLightScattered
        }

        public func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                let wiLocal = worldToLocal(world: wiWorld)
                let woLocal = worldToLocal(world: woWorld)
                if woLocal.z == 0 { return 0 }
                return probabilityDensityLocal(wo: woLocal, wi: wiLocal)
        }

        public func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
        {
                let woLocal = worldToLocal(world: woWorld)
                let bsdfSample = sampleLocal(wo: woLocal, u: u)
                let wiWorld = localToWorld(local: bsdfSample.incoming)
                return (
                        BsdfSample(bsdfSample.estimate, wiWorld, bsdfSample.probabilityDensity),
                        isTransmissive
                )
        }

        public func evaluateDistributionFunction(wo: Vector, wi: Vector, normal: Normal) -> RgbSpectrum {
                let reflected = evaluateWorld(wo: wo, wi: wi)
                let dot = absDot(wi, Vector(normal: normal))
                let scatter = reflected * dot
                return scatter
        }

        public func sampleDistributionFunction(wo: Vector, normal: Normal, sampler: inout Sampler)
                -> BsdfSample
        {
                var (bsdfSample, _) = sampleWorld(wo: wo, u: sampler.get3D())
                bsdfSample.estimate *= absDot(bsdfSample.incoming, normal)
                return bsdfSample
        }

        public func evaluateProbabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                return probabilityDensityWorld(wo: wo, wi: wi)
        }
}
