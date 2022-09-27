import Foundation

///        Bidirectional Scattering Distribution Function
///        Describes how light is scattered by a surface.
struct BSDF {

        init() {
                ng = Normal()
                ns = Normal()
                ss = up
                ts = up
        }

        init(interaction: Interaction) {
                ng = interaction.normal
                ns = interaction.shadingNormal
                ss = normalized(interaction.dpdu)
                ts = cross(Vector(normal: ns), ss)
        }

        mutating func add(bxdf: BxDF) {
                bxdfs.append(bxdf)
        }

        func evaluate(wo woWorld: Vector, wi wiWorld: Vector) -> Spectrum {
                var totalLightScattered = black
                let woLocal = worldToLocal(world: woWorld)
                let wiLocal = worldToLocal(world: wiWorld)
                let reflect = dot(wiWorld, ng) * dot(woWorld, ng) > 0
                //print("evaluate reflect wo wi: ", reflect, woWorld, wiWorld)
                for bxdf in bxdfs {
                        if reflect && bxdf.isReflective {
                                //print("reflect and reflective")
                                totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                        }
                        if !reflect && bxdf.isTransmissive {
                                //print("not reflect and transmissive")
                                totalLightScattered += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                        }
                }
                return totalLightScattered
        }

        func albedo() -> Spectrum {
                var total = black
                for bxdf in bxdfs {
                        total += bxdf.albedo()
                }
                return total / FloatX(bxdfs.count)
        }

        private func worldToLocal(world: Vector) -> Vector {
                return normalized(Vector(x: dot(world, ss), y: dot(world, ts), z: dot(world, ns)))
        }

        private func localToWorld(local: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: ss.x * local.x + ts.x * local.y + ns.x * local.z,
                                y: ss.y * local.x + ts.y * local.y + ns.y * local.z,
                                z: ss.z * local.x + ts.z * local.y + ns.z * local.z))
        }

        func sample(wo woWorld: Vector, u: Point2F) throws -> (
                L: Spectrum, wi: Vector, pdf: FloatX, isTransmissive: Bool
        ) {
                guard let randomBxdf = bxdfs.randomElement() else {
                        warning("BxDF count zero!")
                        return (black, up, 0, false)
                }
                let woLocal = worldToLocal(world: woWorld)
                var (estimate, wiLocal, density) = randomBxdf.sample(wo: woLocal, u: u)
                //print("randomBxdf, estimate, density: ", randomBxdf, estimate, density)
                let wiWorld = localToWorld(local: wiLocal)
                let reflect = dot(wiWorld, ng) * dot(woWorld, ng) > 0
                for bxdf in bxdfs {
                        //print("sample bxdf")
                        if reflect && bxdf.isReflective {
                                if bxdf !== randomBxdf {
                                        estimate += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                                        density += bxdf.probabilityDensity(wo: woLocal, wi: wiLocal)
                                        //print("density now: ", density)
                                }
                        }
                        if !reflect && bxdf.isTransmissive {
                                if bxdf !== randomBxdf {
                                        estimate += bxdf.evaluate(wo: woLocal, wi: wiLocal)
                                        density += bxdf.probabilityDensity(wo: woLocal, wi: wiLocal)
                                        //print("density now: ", density)
                                }
                        }
                }
                //print("BSDF wo wi density: ", woWorld, wiWorld, density)
                return (estimate, wiWorld, density, randomBxdf.isTransmissive)
        }

        func probabilityDensity(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                guard bxdfs.count >= 1 else {
                        warning("BxDF count zero!")
                        return 0
                }
                let wiLocal = worldToLocal(world: wiWorld)
                let woLocal = worldToLocal(world: woWorld)
                if woLocal.z == 0 { return 0 }
                var density: FloatX = 0.0
                for bxdf in bxdfs {
                        density += bxdf.probabilityDensity(wo: woLocal, wi: wiLocal)
                }
                return density
        }

        var bxdfs = [BxDF]()
        var ng = Normal()
        var ns = Normal()
        var ss = up
        var ts = up
        var eta: FloatX = 1.0
}
