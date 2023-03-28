struct DielectricBsdf: BxDF {

        private func backfacing(wi: Vector, wo: Vector, half: Vector) -> Bool {
                return dot(half, wi) * cosTheta(wi) < 0 || dot(half, wo) * cosTheta(wo) < 0
        }

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return black
                }
                let cosThetaO = cosTheta(wo)
                let cosThetaI = cosTheta(wi)
                let reflect = cosThetaI * cosThetaO > 0
                var etap: FloatX = 1
                if !reflect {
                        etap = cosThetaO > 0 ? refractiveIndex : (1 / refractiveIndex)
                }
                var half = wi * etap + wo
                if cosThetaI == 0 || cosThetaO == 0 || lengthSquared(half) == 0 {
                        return black
                }
                half = faceforward(vector: normalized(half), comparedTo: Normal(x: 0, y: 0, z: 1))
                if backfacing(wi: wi, wo: wo, half: half) {
                        return black
                }
                let fresnelDielectric = FresnelDielectric(refractiveIndex: refractiveIndex)
                let fresnel = fresnelDielectric.evaluate(cosTheta: dot(wo, half))
                if reflect {
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                        let enumerator = differentialArea * visibleFraction
                        let denominator = abs(4 * cosThetaI * cosThetaO)
                        return RGBSpectrum(intensity: enumerator / denominator)
                } else {
                        let denominator = square(dot(wi, half) + dot(wo, half) / etap) * cosThetaI * cosThetaO
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                        let oneMinusFresnel = white - fresnel
                        let absDotI = abs(dot(wi, half))
                        let absDotO = abs(dot(wo, half))
                        let absDotIO = absDotI * absDotO
                        var enumerator = differentialArea * oneMinusFresnel * visibleFraction * absDotIO
                        // radiance transport
                        enumerator /= square(etap)
                        return enumerator / denominator
                }
        }

        func sample(wo: Vector, u: Point2F) -> (RGBSpectrum, Vector, FloatX) {

                unimplemented()
                //let fresnelDielectric = FresnelDielectric(etaI: etaA, etaT: etaB)
                //let fresnel = fresnelDielectric.evaluate(cosTheta: cosTheta(wo))
                //if u.x < fresnel.average() {
                //        return reflective(wo: wo, fresnel: fresnel.average())
                //} else {
                //        return transmittive(wo: wo, fresnel: fresnel.average())
                //}
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                unimplemented()
        }

        func albedo() -> RGBSpectrum { return white }

        let distribution: MicrofacetDistribution

        let refractiveIndexVacuum: FloatX = 1

        // eta in PBRT
        // Vacuum: 1
        // Water: 1.3
        // Glass: 1.52
        let refractiveIndex: FloatX
}
