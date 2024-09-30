struct SurfaceInteraction: Interaction, Sendable {

        func evaluateDistributionFunction(wi: Vector) async -> RgbSpectrum {
                let reflected = await bsdf.evaluateWorld(wo: wo, wi: wi)
                let dot = absDot(wi, Vector(normal: shadingNormal))
                let scatter = reflected * dot
                return scatter
        }

        func sampleDistributionFunction(sampler: Sampler) async -> BsdfSample {
                var (bsdfSample, _) = await bsdf.sampleWorld(wo: wo, u: sampler.get3D())
                bsdfSample.estimate *= absDot(bsdfSample.incoming, shadingNormal)
                return bsdfSample
        }

        func evaluateProbabilityDensity(wi: Vector) -> FloatX {
                return bsdf.probabilityDensityWorld(wo: wo, wi: wi)
        }

        @MainActor
        mutating func setBsdf() {
                bsdf = materials[material].getBsdf(interaction: self)
        }

        var valid = false
        var position = Point()
        var normal = Normal()
        var shadingNormal = Normal()
        var wo = Vector()
        var dpdu = Vector()
        var uv = Point2F()
        var faceIndex = 0
        var areaLight: AreaLight? = nil
        var material: MaterialIndex = -1
        var mediumInterface: MediumInterface? = nil
        var bsdf: GlobalBsdf = DummyBsdf()
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
