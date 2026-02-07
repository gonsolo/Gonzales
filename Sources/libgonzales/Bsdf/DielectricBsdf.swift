public struct DielectricBsdf: GlobalBsdf {

        public let distribution: any MicrofacetDistribution
        public let refractiveIndexVacuum: FloatX = 1
        // eta in PBRT
        // Vacuum: 1
        // Water: 1.3
        // Glass: 1.52
        public let refractiveIndex: FloatX
        public let bsdfFrame: BsdfFrame

        public init(distribution: any MicrofacetDistribution, refractiveIndex: FloatX, bsdfFrame: BsdfFrame) {
                self.distribution = distribution
                self.refractiveIndex = refractiveIndex
                self.bsdfFrame = bsdfFrame
        }
}

extension DielectricBsdf {

        private func backfacing(wi: Vector, wo: Vector, half: Vector) -> Bool {
                return dot(half, wi) * cosTheta(wi) < 0 || dot(half, wo) * cosTheta(wo) < 0
        }

        public func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {
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
                halfVector: Vector,
                reflected: FloatX,
                probabilityReflected: FloatX
        ) -> BsdfSample {
                let wi = reflect(vector: wo, by: halfVector)
                guard sameHemisphere(wo, wi) else {
                        return BsdfSample()
                }
                let probabilityDensity =
                        distribution.probabilityDensity(wo: wo, half: halfVector)
                        / (4 * absDot(wo, halfVector)) * probabilityReflected

                let differentialArea = distribution.differentialArea(withNormal: halfVector)
                let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                let enumerator = differentialArea * visibleFraction * reflected
                let denominator = 4 * cosTheta(wi) * cosTheta(wo)
                let estimate = RgbSpectrum(intensity: enumerator / denominator)
                return BsdfSample(estimate, wi, probabilityDensity)
        }

        private func sampleRoughTransmission(
                wo: Vector,
                halfVector: Vector,
                transmitted: FloatX,
                probabilityTransmitted: FloatX
        ) -> BsdfSample {
                guard let (wi, etap) = refract(wi: wo, normal: Normal(halfVector), eta: refractiveIndex) else {
                        return BsdfSample()
                }
                if sameHemisphere(wo, wi) || wi.z == 0 {
                        return BsdfSample()
                }
                let denom = square(dot(wi, halfVector) + dot(wo, halfVector) / etap)
                let dwmDwi = absDot(wi, halfVector) / denom
                let probabilityDensity =
                        distribution.probabilityDensity(wo: wo, half: halfVector) * dwmDwi * probabilityTransmitted
                let differentialArea = distribution.differentialArea(withNormal: halfVector)
                let visibleFraction = distribution.visibleFraction(from: wo, and: wi)
                var estimate = RgbSpectrum(
                        intensity:
                                transmitted * differentialArea * visibleFraction
                                * abs(dot(wi, halfVector) * dot(wo, halfVector) / cosTheta(wi) * cosTheta(wo) * denom))
                // transport mode radiance
                estimate /= square(etap)
                return BsdfSample(estimate, wi, probabilityDensity)
        }

        private func sampleRough(wo: Vector, u: ThreeRandomVariables) -> BsdfSample {
                let halfVector = distribution.sampleHalfVector(wo: wo, u: (u.0, u.1))
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, halfVector),
                        refractiveIndex: refractiveIndex)
                let transmitted = 1 - reflected
                let probabilityReflected = reflected / (reflected + transmitted)
                let probabilityTransmitted = transmitted / (reflected + transmitted)
                if u.2 < probabilityReflected {
                        return sampleRoughReflection(
                                wo: wo,
                                halfVector: halfVector,
                                reflected: reflected,
                                probabilityReflected: probabilityReflected)
                } else {
                        return sampleRoughTransmission(
                                wo: wo,
                                halfVector: halfVector,
                                transmitted: transmitted,
                                probabilityTransmitted: probabilityTransmitted)
                }
        }

        public func sampleLocal(wo: Vector, u: ThreeRandomVariables) -> BsdfSample {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return sampleSpecular(wo: wo, u: u)
                } else {
                        return sampleRough(wo: wo, u: u)
                }
        }

        public func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX {
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
                var halfVector = wi * etap + wo
                if cosThetaO.isZero || cosThetaI.isZero || lengthSquared(halfVector).isZero {
                        return 0
                }
                halfVector = faceforward(vector: normalized(halfVector), comparedTo: Normal(x: 0, y: 0, z: 1))
                if dot(halfVector, wi) * cosThetaI < 0 || dot(halfVector, wo) * cosThetaO < 0 {
                        return 0
                }
                let fresnelR = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, halfVector),
                        refractiveIndex: refractiveIndex)
                let fresnelT = 1 - fresnelR
                let probabilityReflected = fresnelR
                let probabilityTransmitted = fresnelT
                var pdf: FloatX = 0
                if reflect {
                        let dPdf: FloatX = distribution.probabilityDensity(wo: wo, half: halfVector)
                        let four: FloatX = 4 * absDot(wo, halfVector)
                        let p: FloatX = probabilityReflected / (probabilityReflected + probabilityTransmitted)
                        pdf = dPdf / four * p
                } else {
                        let denom = square(dot(wi, halfVector) + dot(wo, halfVector) / etap)
                        let dwmDwi = absDot(wi, halfVector) / denom
                        pdf = distribution.probabilityDensity(wo: wo, half: halfVector) * dwmDwi
                                * probabilityTransmitted / (probabilityReflected + probabilityTransmitted)
                }
                return pdf
        }

        public func albedo() -> RgbSpectrum { return white }

        var isSpecular: Bool {
                return distribution.isSmooth
        }

}
