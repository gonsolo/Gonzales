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

        private func backfacing(incident: Vector, outgoing: Vector, half: Vector) -> Bool {
                return dot(half, incident) * cosTheta(incident) < 0 || dot(half, outgoing) * cosTheta(outgoing) < 0
        }

        public func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return black
                }
                let cosThetaO = cosTheta(outgoing)
                let cosThetaI = cosTheta(incident)
                let reflect = cosThetaI * cosThetaO > 0
                var etap: FloatX = 1
                if !reflect {
                        etap = cosThetaO > 0 ? refractiveIndex : (1 / refractiveIndex)
                }
                var half = incident * etap + outgoing
                if cosThetaI == 0 || cosThetaO == 0 || lengthSquared(half) == 0 {
                        return black
                }
                half = faceforward(vector: normalized(half), comparedTo: Normal(x: 0, y: 0, z: 1))
                if backfacing(incident: incident, outgoing: outgoing, half: half) {
                        return black
                }
                let fresnelReflected = FresnelDielectric.reflected(
                        cosThetaI: dot(outgoing, half),
                        refractiveIndex: refractiveIndex
                )
                if reflect {
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: outgoing, and: incident)
                        let enumerator = differentialArea * visibleFraction
                        let denominator = abs(4 * cosThetaI * cosThetaO)
                        let estimate = RgbSpectrum(intensity: enumerator / denominator)
                        return estimate
                } else {
                        let denominator = square(dot(incident, half) + dot(outgoing, half) / etap)
                                * cosThetaI * cosThetaO
                        let differentialArea = distribution.differentialArea(withNormal: half)
                        let visibleFraction = distribution.visibleFraction(from: outgoing, and: incident)
                        let fresnelTransmitted = 1 - fresnelReflected
                        let absDotI = abs(dot(incident, half))
                        let absDotO = abs(dot(outgoing, half))
                        let absDotIO = absDotI * absDotO
                        var enumerator = differentialArea * fresnelTransmitted * visibleFraction * absDotIO
                        // radiance transport
                        enumerator /= square(etap)
                        let estimate = RgbSpectrum(intensity: enumerator / denominator)
                        return estimate
                }
        }

        private func sampleSpecularReflection(outgoing: Vector, reflected: FloatX, probabilityReflected: FloatX)
                -> BsdfSample
        {
                let incident = mirror(outgoing)
                let estimate = RgbSpectrum(intensity: reflected / absCosTheta(incident))
                return BsdfSample(estimate, incident, probabilityReflected)
        }

        private func sampleSpecularTransmission(
                outgoing: Vector,
                transmitted: FloatX,
                probabilityTransmitted: FloatX
        ) -> BsdfSample {
                let upNormal = Normal(x: 0, y: 0, z: 1)
                guard let (incident, etap) = refract(incident: outgoing, normal: upNormal, eta: refractiveIndex) else {
                        return BsdfSample()
                }
                var estimate = RgbSpectrum(intensity: transmitted / absCosTheta(incident))
                // transport mode radiance
                estimate /= square(etap)
                return BsdfSample(estimate, incident, probabilityTransmitted)
        }

        private func sampleSpecular(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: cosTheta(outgoing),
                        refractiveIndex: refractiveIndex)
                let transmitted = 1 - reflected
                let probabilityReflected = reflected / (reflected + transmitted)
                let probabilityTransmitted = transmitted / (reflected + transmitted)
                if uSample.0 < probabilityReflected {
                        return sampleSpecularReflection(
                                outgoing: outgoing,
                                reflected: reflected,
                                probabilityReflected: probabilityReflected)
                } else {
                        return sampleSpecularTransmission(
                                outgoing: outgoing,
                                transmitted: transmitted,
                                probabilityTransmitted: probabilityTransmitted)
                }
        }

        private func sampleRoughReflection(
                outgoing: Vector,
                halfVector: Vector,
                reflected: FloatX,
                probabilityReflected: FloatX
        ) -> BsdfSample {
                let incident = reflect(vector: outgoing, by: halfVector)
                guard sameHemisphere(outgoing, incident) else {
                        return BsdfSample()
                }
                let probabilityDensity =
                        distribution.probabilityDensity(outgoing: outgoing, half: halfVector)
                        / (4 * absDot(outgoing, halfVector)) * probabilityReflected

                let differentialArea = distribution.differentialArea(withNormal: halfVector)
                let visibleFraction = distribution.visibleFraction(from: outgoing, and: incident)
                let enumerator = differentialArea * visibleFraction * reflected
                let denominator = 4 * cosTheta(incident) * cosTheta(outgoing)
                let estimate = RgbSpectrum(intensity: enumerator / denominator)
                return BsdfSample(estimate, incident, probabilityDensity)
        }

        private func sampleRoughTransmission(
                outgoing: Vector,
                halfVector: Vector,
                transmitted: FloatX,
                probabilityTransmitted: FloatX
        ) -> BsdfSample {
                guard let (incident, etap) = refract(
                        incident: outgoing, normal: Normal(halfVector), eta: refractiveIndex)
                else {
                        return BsdfSample()
                }
                if sameHemisphere(outgoing, incident) || incident.z == 0 {
                        return BsdfSample()
                }
                let denom = square(dot(incident, halfVector) + dot(outgoing, halfVector) / etap)
                let dwmDwi = absDot(incident, halfVector) / denom
                let probabilityDensity =
                        distribution.probabilityDensity(outgoing: outgoing, half: halfVector) * dwmDwi
                                * probabilityTransmitted
                let differentialArea = distribution.differentialArea(withNormal: halfVector)
                let visibleFraction = distribution.visibleFraction(from: outgoing, and: incident)
                var estimate = RgbSpectrum(
                        intensity:
                                transmitted * differentialArea * visibleFraction
                                * abs(
                                        dot(incident, halfVector) * dot(outgoing, halfVector) / cosTheta(incident)
                                                * cosTheta(outgoing) * denom))
                // transport mode radiance
                estimate /= square(etap)
                return BsdfSample(estimate, incident, probabilityDensity)
        }

        private func sampleRough(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                let halfVector = distribution.sampleHalfVector(outgoing: outgoing, uSample: (uSample.0, uSample.1))
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: dot(outgoing, halfVector),
                        refractiveIndex: refractiveIndex)
                let transmitted = 1 - reflected
                let probabilityReflected = reflected / (reflected + transmitted)
                let probabilityTransmitted = transmitted / (reflected + transmitted)
                if uSample.2 < probabilityReflected {
                        return sampleRoughReflection(
                                outgoing: outgoing,
                                halfVector: halfVector,
                                reflected: reflected,
                                probabilityReflected: probabilityReflected)
                } else {
                        return sampleRoughTransmission(
                                outgoing: outgoing,
                                halfVector: halfVector,
                                transmitted: transmitted,
                                probabilityTransmitted: probabilityTransmitted)
                }
        }

        public func sampleLocal(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return sampleSpecular(outgoing: outgoing, uSample: uSample)
                } else {
                        return sampleRough(outgoing: outgoing, uSample: uSample)
                }
        }

        public func probabilityDensityLocal(outgoing: Vector, incident: Vector) -> FloatX {
                if refractiveIndex == refractiveIndexVacuum || isSpecular {
                        return 0
                }
                let cosThetaO = cosTheta(outgoing)
                let cosThetaI = cosTheta(incident)
                let reflect = cosThetaI * cosThetaO > 0
                var etap: FloatX = 1
                if !reflect {
                        etap = cosThetaO > 0 ? refractiveIndex : 1 / refractiveIndex
                }
                var halfVector = incident * etap + outgoing
                if cosThetaO.isZero || cosThetaI.isZero || lengthSquared(halfVector).isZero {
                        return 0
                }
                halfVector = faceforward(vector: normalized(halfVector), comparedTo: Normal(x: 0, y: 0, z: 1))
                if dot(halfVector, incident) * cosThetaI < 0 || dot(halfVector, outgoing) * cosThetaO < 0 {
                        return 0
                }
                let fresnelR = FresnelDielectric.reflected(
                        cosThetaI: dot(outgoing, halfVector),
                        refractiveIndex: refractiveIndex)
                let fresnelT = 1 - fresnelR
                let probabilityReflected = fresnelR
                let probabilityTransmitted = fresnelT
                var pdf: FloatX = 0
                if reflect {
                        let dPdf: FloatX = distribution.probabilityDensity(outgoing: outgoing, half: halfVector)
                        let four: FloatX = 4 * absDot(outgoing, halfVector)
                        let probability: FloatX = probabilityReflected / (probabilityReflected + probabilityTransmitted)
                        pdf = dPdf / four * probability
                } else {
                        let denom = square(dot(incident, halfVector) + dot(outgoing, halfVector) / etap)
                        let dwmDwi = absDot(incident, halfVector) / denom
                        pdf = distribution.probabilityDensity(outgoing: outgoing, half: halfVector) * dwmDwi
                                * probabilityTransmitted / (probabilityReflected + probabilityTransmitted)
                }
                return pdf
        }

        public func albedo() -> RgbSpectrum { return white }

        var isSpecular: Bool {
                return distribution.isSmooth
        }

}
