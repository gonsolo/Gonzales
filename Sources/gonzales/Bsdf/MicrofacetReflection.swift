struct MicrofacetReflection: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                let cosThetaO = absCosTheta(wo)
                let cosThetaI = absCosTheta(wi)
                var half = wo + wi
                if cosThetaO == 0 || cosThetaI == 0 { return RGBSpectrum() }
                guard !half.isZero else { return black }
                half.normalize()
                let f = fresnel.evaluate(cosTheta: dot(wi, half))
                let area = distribution.differentialArea(withNormal: half)
                let visible = distribution.visibleFraction(from: wo, and: wi)
                return reflectance * area * visible * f / (4 * cosThetaI * cosThetaO)
        }

        func sample(wo: Vector, u: Point2F) -> (RGBSpectrum, Vector, FloatX) {
                guard !wo.z.isZero else {
                        return (black, nullVector, 0.0)
                }
                let half = distribution.sampleHalfVector(wo: wo, u: u)
                let wi = reflect(vector: wo, by: half)
                let radiance = evaluate(wo: wo, wi: wi)
                let density = probabilityDensity(wo: wo, wi: wi)
                return (radiance, wi, density)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                guard sameHemisphere(wo, wi) else { return 0 }
                let half = normalized(wo + wi)
                return distribution.pdf(wo: wo, half: half) / (4 * dot(wo, half))
        }

        func albedo() -> RGBSpectrum { return white }

        var reflectance: RGBSpectrum
        var distribution: MicrofacetDistribution
        var fresnel: Fresnel
}
