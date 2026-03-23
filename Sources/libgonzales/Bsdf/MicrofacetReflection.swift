struct MicrofacetReflection: FramedBsdf {
        var isSpecular: Bool { return false }

        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                let cosThetaO = absCosTheta(outgoing)
                let cosThetaI = absCosTheta(incident)
                var half = outgoing + incident
                if cosThetaO == 0 || cosThetaI == 0 { return RgbSpectrum() }
                guard !half.isZero else { return black }
                half.normalize()
                let fresnelValue = fresnel.evaluate(cosTheta: dot(incident, half))
                let area = distribution.differentialArea(withNormal: half)
                let visible = distribution.visibleFraction(from: outgoing, and: incident)
                return reflectance * area * visible * fresnelValue / (4 * cosThetaI * cosThetaO)
        }

        func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                guard !outgoing.z.isZero else {
                        return BsdfSample()
                }
                let half = distribution.sampleHalfVector(outgoing: outgoing, uSample: (uSample.0, uSample.1))
                let incident = reflect(vector: outgoing, by: half)
                let radiance = evaluate(outgoing: outgoing, incident: incident)
                let density = probabilityDensity(outgoing: outgoing, incident: incident)
                return BsdfSample(radiance, incident, density)
        }

        func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                guard sameHemisphere(outgoing, incident) else { return 0 }
                let half = normalized(outgoing + incident)
                return distribution.probabilityDensity(outgoing: outgoing, half: half) / (4 * dot(outgoing, half))
        }

        func albedo() -> RgbSpectrum { return white }

        var reflectance: RgbSpectrum
        var distribution: any MicrofacetDistribution
        var fresnel: any Fresnel

        let bsdfFrame: BsdfFrame
}
