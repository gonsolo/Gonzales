///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
protocol GlobalBsdf: LocalBsdf {
        func albedoWorld() -> RGBSpectrum
        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RGBSpectrum
        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX
        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                throws -> (bsdfSample: BsdfSample, isTransmissive: Bool)

        func worldToLocal(world: Vector) -> Vector
        func localToWorld(local: Vector) -> Vector
        func isReflecting(wi: Vector, wo: Vector) -> Bool
}

extension GlobalBsdf {

        func albedoWorld() -> RGBSpectrum {
                return albedoLocal()
        }

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
                throws -> (bsdfSample: BsdfSample, isTransmissive: Bool)
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

struct BsdfGeometry {

        init() {
                geometricNormal = Normal()
                frame = ShadingFrame()
        }

        init(geometricNormal: Normal, frame: ShadingFrame) {
                self.geometricNormal = geometricNormal
                self.frame = frame
        }

        init(interaction: Interaction) {
                let frame = ShadingFrame(
                        x: Vector(normal: interaction.shadingNormal),
                        y: normalized(interaction.dpdu)
                )
                self.geometricNormal = interaction.normal
                self.frame = frame
        }

        func isReflecting(wi: Vector, wo: Vector) -> Bool {
                return dot(wi, geometricNormal) * dot(wo, geometricNormal) > 0
        }

        let geometricNormal: Normal
        let frame: ShadingFrame
}
