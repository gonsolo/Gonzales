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
                let fresnelReflected = FresnelDielectric.reflected(
                        cosThetaI: dot(wo, half),
                        refractiveIndex: refractiveIndex
                )
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
                        let fresnelTransmitted = 1 - fresnelReflected
                        let absDotI = abs(dot(wi, half))
                        let absDotO = abs(dot(wo, half))
                        let absDotIO = absDotI * absDotO
                        var enumerator = differentialArea * fresnelTransmitted * visibleFraction * absDotIO
                        // radiance transport
                        enumerator /= square(etap)
                        return RGBSpectrum(intensity: enumerator / denominator)
                }
        }

        private func sampleSpecularReflection() -> BSDFSample {
                unimplemented()
        }

        private func sampleSpecularTransmission() -> BSDFSample {
                unimplemented()
        }

        private func sampleSpecular(wo: Vector, u: ThreeRandomVariables) -> BSDFSample {
                let reflected = FresnelDielectric.reflected(
                        cosThetaI: cosTheta(wo),
                        refractiveIndex: refractiveIndex)
                let transmitted = white - reflected
                //if u < fre
                if true {
                        return sampleSpecularReflection()
                } else {
                        return sampleSpecularTransmission()
                }
        }

        private func sampleRoughReflection() -> BSDFSample {
                unimplemented()
        }

        private func sampleRoughTransmission() -> BSDFSample {
                unimplemented()
        }

        private func sampleRough() -> BSDFSample {
                if true {
                        return sampleRoughReflection()
                } else {
                        return sampleRoughTransmission()
                }
        }

        func sample(wo: Vector, u: ThreeRandomVariables) -> BSDFSample {
                if refractiveIndex == refractiveIndexVacuum || distribution.isSmooth {
                        return sampleSpecular(wo: wo, u: u)
                } else {
                        return sampleRough()
                }
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
