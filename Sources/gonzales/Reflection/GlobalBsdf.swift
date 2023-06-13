///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
protocol GlobalBsdf: BsdfFrameProtocol, LocalBsdf {
        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RGBSpectrum
        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX
        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
}

extension GlobalBsdf {

        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RGBSpectrum {
                var totalLightScattered = black
                let woLocal = worldToLocal(world: woWorld)
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

        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                let wiLocal = worldToLocal(world: wiWorld)
                let woLocal = worldToLocal(world: woWorld)
                if woLocal.z == 0 { return 0 }
                return probabilityDensityLocal(wo: woLocal, wi: wiLocal)
        }

        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
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
}
