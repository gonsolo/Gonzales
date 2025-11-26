struct MicrofacetReflection: GlobalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {
                let cosThetaO = absCosTheta(wo)
                let cosThetaI = absCosTheta(wi)
                var half = wo + wi
                if cosThetaO == 0 || cosThetaI == 0 { return RgbSpectrum() }
                guard !half.isZero else { return black }
                half.normalize()
                let f = fresnel.evaluate(cosTheta: dot(wi, half))
                let area = distribution.differentialArea(withNormal: half)
                let visible = distribution.visibleFraction(from: wo, and: wi)
                return reflectance * area * visible * f / (4 * cosThetaI * cosThetaO)
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                guard !wo.z.isZero else {
                        return BsdfSample()
                }
                let half = distribution.sampleHalfVector(wo: wo, u: (u.0, u.1))
                let wi = reflect(vector: wo, by: half)
                let radiance = evaluateLocal(wo: wo, wi: wi)
                let density = probabilityDensityLocal(wo: wo, wi: wi)
                return BsdfSample(radiance, wi, density)
        }

        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX {
                guard sameHemisphere(wo, wi) else { return 0 }
                let half = normalized(wo + wi)
                return distribution.probabilityDensity(wo: wo, half: half) / (4 * dot(wo, half))
        }

        func albedo() -> RgbSpectrum { return white }

        var reflectance: RgbSpectrum
        var distribution: any MicrofacetDistribution
        var fresnel: any Fresnel

        let bsdfFrame: BsdfFrame
}
