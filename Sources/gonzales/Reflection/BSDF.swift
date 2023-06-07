///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
struct BSDF {

        init() {
                bxdf = DiffuseBsdf(reflectance: black)
        }

        init(interaction: Interaction) {
                bxdf = DiffuseBsdf(reflectance: black)
                geometricNormal = interaction.normal
                let ss = normalized(interaction.dpdu)
                let ts = cross(Vector(normal: interaction.shadingNormal), ss)
                frame = CoordinateFrame(x: Vector(normal: interaction.shadingNormal), y: ss, z: ts)
        }

        mutating func set(bxdf: BxDF) {
                self.bxdf = bxdf
        }

        func evaluate(wo woWorld: Vector, wi wiWorld: Vector) -> RGBSpectrum {
                var totalLightScattered = black
                let woLocal = worldToLocal(world: woWorld)
                let wiLocal = worldToLocal(world: wiWorld)
                let reflect = dot(wiWorld, geometricNormal) * dot(woWorld, geometricNormal) > 0
                if reflect && bxdf.isReflective {
                        totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                }
                if !reflect && bxdf.isTransmissive {
                        totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                }
                return totalLightScattered
        }

        func albedo() -> RGBSpectrum {
                return bxdf.albedo()
        }

        private func worldToLocal(world: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: dot(world, frame.y),
                                y: dot(world, frame.z),
                                z: dot(world, frame.x)))
        }

        private func localToWorld(local: Vector) -> Vector {
                let x = frame.y.x * local.x + frame.z.x * local.y + frame.x.x * local.z
                let y = frame.y.y * local.x + frame.z.y * local.y + frame.x.y * local.z
                let z = frame.y.z * local.x + frame.z.z * local.y + frame.x.z * local.z
                return normalized(Vector(x: x, y: y, z: z))
        }

        func sample(wo woWorld: Vector, u: ThreeRandomVariables)
                throws -> (bsdfSample: BSDFSample, isTransmissive: Bool)
        {
                let woLocal = worldToLocal(world: woWorld)
                let bsdfSample = bxdf.sample(wo: woLocal, u: u)
                let wiWorld = localToWorld(local: bsdfSample.incoming)
                return (
                        BSDFSample(bsdfSample.estimate, wiWorld, bsdfSample.probabilityDensity),
                        bxdf.isTransmissive
                )
        }

        func probabilityDensity(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                let wiLocal = worldToLocal(world: wiWorld)
                let woLocal = worldToLocal(world: woWorld)
                if woLocal.z == 0 { return 0 }
                return bxdf.probabilityDensity(wo: woLocal, wi: wiLocal)
        }

        var bxdf: BxDF
        var geometricNormal = Normal()
        var frame = CoordinateFrame()
}
