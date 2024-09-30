struct DielectricBsdf: GlobalBsdf {

        private func backfacing(wi: Vector, wo: Vector, half: Vector) -> Bool {
                return dot(half, wi) * cosTheta(wi) < 0 || dot(half, wo) * cosTheta(wo) < 0
        }

        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {
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
                let fresnelReflected = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, half),
                        refractiveIndex: refractiveIndex
                )
                if reflect {
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                        let enumerator = differentialArea * visibleFraction
                        let denominator = abs(4 * cosThetaI * cosThetaO)
                        let estimate = RgbSpectrum(intensity: enumerator / denominator)
                        return estimate
                } else {
                        let denominator = square(dot(wi, half) + dot(wo, half) / etap) * cosThetaI * cosThetaO
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                        let fresnelTransmitted = 1 - fresnelReflected
                        let absDotI = abs(dot(wi, half))
                        let absDotO = abs(dot(wo, half))
                        let absDotIO = absDotI * absDotO
                        var enumerator = differentialArea * fresnelTransmitted * visibleFraction * absDotIO
                        // radiance transport
                        enumerator /= square(etap)
                        let estimate = RgbSpectrum(intensity: enumerator / denominator)
                        return estimate
                }
        }

        private func sampleSpecularReflection(wo: Vector, reflected: FloatX, probabilityReflected: FloatX)
                -> BsdfSample
        {
                let wi = mirror(wo)
                let estimate = RgbSpectrum(intensity: reflected / absCosTheta(wi))
                return BsdfSample(estimate, wi, probabilityReflected)
        }

        private func sampleSpecularTransmission(
                wo: Vector,
                transmitted: FloatX,
                probabilityTransmitted: FloatX
        ) -> BsdfSample {
                let up = Normal(x: 0, y: 0, z: 1)
                guard let (wi, etap) = refract(wi: wo, normal: up, eta: refractiveIndex) else {
                        return BsdfSample()
                }
                var estimate = RgbSpectrum(intensity: transmitted / absCosTheta(wi))
                // transport mode radiance
                estimate /= square(etap)
                return BsdfSample(estimate, wi, probabilityTransmitted)
        }

        private func sampleSpecular(wo: Vector, u: ThreeRandomVariables) -> BsdfSample {
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: cosTheta(wo),
                        refractiveIndex: refractiveIndex)
                let transmitted = 1 - reflected
                let probabilityReflected = reflected / (reflected + transmitted)
                let probabilityTransmitted = transmitted / (reflected + transmitted)
                if u.0 < probabilityReflected {
                        return sampleSpecularReflection(
                                wo: wo,
                                reflected: reflected,
                                probabilityReflected: probabilityReflected)
                } else {
                        return sampleSpecularTransmission(
                                wo: wo,
                                transmitted: transmitted,
                                probabilityTransmitted: probabilityTransmitted)
                }
        }

        private func sampleRoughReflection(
                wo: Vector,
                wm: Vector,
                reflected: FloatX,
                probabilityReflected: FloatX
        ) -> BsdfSample {
                let wi = reflect(vector: wo, by: wm)
                guard sameHemisphere(wo, wi) else {
                        return BsdfSample()
                }
                let probabilityDensity =
                        distribution.probabilityDensity(wo: wo, half: wm)
                        / (4 * absDot(wo, wm)) * probabilityReflected

                let differentialArea = distribution.differentialArea(withNormal: wm)
                let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                let enumerator = differentialArea * visibleFraction * reflected
                let denominator = 4 * cosTheta(wi) * cosTheta(wo)
                let estimate = RgbSpectrum(intensity: enumerator / denominator)
                return BsdfSample(estimate, wi, probabilityDensity)
        }

        private func sampleRoughTransmission(
                wo: Vector,
                wm: Vector,
                transmitted: FloatX,
                probabilityTransmitted: FloatX
        ) -> BsdfSample {
                guard let (wi, etap) = refract(wi: wo, normal: Normal(wm), eta: refractiveIndex) else {
                        return BsdfSample()
                }
                if sameHemisphere(wo, wi) || wi.z == 0 {
                        return BsdfSample()
                }
                let denom = square(dot(wi, wm) + dot(wo, wm) / etap)
                let dwmDwi = absDot(wi, wm) / denom
                let probabilityDensity =
                        distribution.probabilityDensity(wo: wo, half: wm) * dwmDwi * probabilityTransmitted
                let differentialArea = distribution.differentialArea(withNormal: wm)
                let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                var estimate = RgbSpectrum(
                        intensity:
                                transmitted * differentialArea * visibleFraction
                                * abs(dot(wi, wm) * dot(wo, wm) / cosTheta(wi) * cosTheta(wo) * denom))
                // transport mode radiance
                estimate /= square(etap)
                return BsdfSample(estimate, wi, probabilityDensity)
        }

        private func sampleRough(wo: Vector, u: ThreeRandomVariables) -> BsdfSample {
                let wm = distribution.sampleHalfVector(wo: wo, u: (u.0, u.1))
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, wm),
                        refractiveIndex: refractiveIndex)
                let transmitted = 1 - reflected
                let probabilityReflected = reflected / (reflected + transmitted)
                let probabilityTransmitted = transmitted / (reflected + transmitted)
                if u.2 < probabilityReflected {
                        return sampleRoughReflection(
                                wo: wo,
                                wm: wm,
                                reflected: reflected,
                                probabilityReflected: probabilityReflected)
                } else {
                        return sampleRoughTransmission(
                                wo: wo,
                                wm: wm,
                                transmitted: transmitted,
                                probabilityTransmitted: probabilityTransmitted)
                }
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return sampleSpecular(wo: wo, u: u)
                } else {
                        return sampleRough(wo: wo, u: u)
                }
        }

        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX {
                if refractiveIndex == refractiveIndexVacuum || isSpecular {
                        return 0
                }
                let cosThetaO = cosTheta(wo)
                let cosThetaI = cosTheta(wi)
                let reflect = cosThetaI * cosThetaO > 0
                var etap: FloatX = 1
                if !reflect {
                        etap = cosThetaO > 0 ? refractiveIndex : 1 / refractiveIndex
                }
                var wm = wi * etap + wo
                if cosThetaO.isZero || cosThetaI.isZero || lengthSquared(wm).isZero {
                        return 0
                }
                wm = faceforward(vector: normalized(wm), comparedTo: Normal(x: 0, y: 0, z: 1))
                if dot(wm, wi) * cosThetaI < 0 || dot(wm, wo) * cosThetaO < 0 {
                        return 0
                }
                let fresnelR = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, wm),
                        refractiveIndex: refractiveIndex)
                let fresnelT = 1 - fresnelR
                let pr = fresnelR
                let pt = fresnelT
                var pdf: FloatX = 0
                if reflect {
                        let dPdf: FloatX = distribution.probabilityDensity(wo: wo, half: wm)
                        let four: FloatX = 4 * absDot(wo, wm)
                        let p: FloatX = pr / (pr + pt)
                        pdf = dPdf / four * p
                } else {
                        let denom = square(dot(wi, wm) + dot(wo, wm) / etap)
                        let dwmDwi = absDot(wi, wm) / denom
                        pdf = distribution.probabilityDensity(wo: wo, half: wm) * dwmDwi * pt / (pr + pt)
                }
                return pdf
        }

        func albedo() -> RgbSpectrum { return white }

        var isSpecular: Bool {
                return distribution.isSmooth
        }

        let distribution: MicrofacetDistribution

        let refractiveIndexVacuum: FloatX = 1

        // eta in PBRT
        // Vacuum: 1
        // Water: 1.3
        // Glass: 1.52
        let refractiveIndex: FloatX

        let bsdfFrame: BsdfFrame
}
