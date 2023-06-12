///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
struct BSDF {

        init() {
                bxdf = DiffuseBsdf(reflectance: black)
                bsdfGeometry = BsdfGeometry()
        }

        init(bxdf: BxDF, interaction: Interaction) {
                self.bxdf = bxdf
                bsdfGeometry = BsdfGeometry(interaction: interaction)
        }

        func albedoWorld() -> RGBSpectrum {
                return bxdf.albedo()
        }

        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RGBSpectrum {
                var totalLightScattered = black
                let woLocal = bsdfGeometry.frame.worldToLocal(world: woWorld)
                let wiLocal = bsdfGeometry.frame.worldToLocal(world: wiWorld)
                let reflect = bsdfGeometry.isReflecting(wi: wiWorld, wo: woWorld)
                if reflect && bxdf.isReflective {
                        totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                }
                if !reflect && bxdf.isTransmissive {
                        totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                }
                return totalLightScattered
        }

        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                let wiLocal = bsdfGeometry.frame.worldToLocal(world: wiWorld)
                let woLocal = bsdfGeometry.frame.worldToLocal(world: woWorld)
                if woLocal.z == 0 { return 0 }
                return bxdf.probabilityDensity(wo: woLocal, wi: wiLocal)
        }

        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                throws -> (bsdfSample: BSDFSample, isTransmissive: Bool)
        {
                let woLocal = bsdfGeometry.frame.worldToLocal(world: woWorld)
                let bsdfSample = bxdf.sample(wo: woLocal, u: u)
                let wiWorld = bsdfGeometry.frame.localToWorld(local: bsdfSample.incoming)
                return (
                        BSDFSample(bsdfSample.estimate, wiWorld, bsdfSample.probabilityDensity),
                        bxdf.isTransmissive
                )
        }

        let bxdf: BxDF
        let bsdfGeometry: BsdfGeometry
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
