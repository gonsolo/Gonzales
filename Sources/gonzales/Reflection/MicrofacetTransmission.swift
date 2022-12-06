struct MicrofacetTransmission: BxDF {

        init(t: Spectrum, distribution: MicrofacetDistribution, eta: (FloatX, FloatX)) {
                self.t = t
                self.distribution = distribution
                self.eta = eta
                self.fresnel = FresnelDielectric(etaI: eta.0, etaT: eta.1)
        }

        private func computeEta(cosTheta: FloatX, eta: (FloatX, FloatX)) -> FloatX {
                if cosTheta > 0 {
                        return self.eta.1 / self.eta.0
                } else {
                        return self.eta.0 / self.eta.1
                }
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {

                if sameHemisphere(wo, wi) {
                        return black
                }
                let cosThetaO = cosTheta(wo)
                let cosThetaI = cosTheta(wi)
                if cosThetaO == 0 || cosThetaI == 0 {
                        return black
                }
                let eta = computeEta(cosTheta: cosThetaO, eta: self.eta)
                var half = normalized(wo + wi * eta)
                if half.z < 0 { half = -half }
                let f = fresnel.evaluate(cosTheta: dot(wo, half))
                let sqrtDenom = dot(wo, half) + eta * dot(wi, half)
                let factor = 1 / eta  // TODO: Transport mode or always radiance with path tracing?
                let d = distribution.differentialArea(withNormal: half)
                let g = distribution.visibleFraction(from: wo, and: wi)
                let termA: Spectrum = (white - f) * t
                let termB: FloatX = d * g * eta * eta
                let termC: FloatX = absDot(wi, half) * absDot(wo, half) * factor * factor
                let termD: FloatX = cosThetaI * cosThetaO * sqrtDenom * sqrtDenom
                let gonzoabs: FloatX = abs(termB * termC / (termD))
                let radiance: Spectrum = termA * gonzoabs
                //print("MicrofacetTransmission D G eta doti doto factor cosThetaI cosThetaO sqrtDenom: ", d, g, eta, absDot(wi, half), absDot(wo, half), factor, cosThetaI, cosThetaO, sqrtDenom)
                return radiance
        }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX) {
                if wo.z == 0 {
                        return (black, nullVector, 0)
                }
                let half = distribution.sampleHalfVector(wo: wo, u: u)
                if dot(wo, half) < 0 {
                        return (black, nullVector, 0)
                }
                let eta = computeEta(cosTheta: -cosTheta(wo), eta: self.eta)
                guard let wi = refract(wi: wo, normal: Normal(half), eta: eta) else {
                        return (black, nullVector, 0)
                }
                let radiance = evaluate(wo: wo, wi: wi)
                let density = probabilityDensity(wo: wo, wi: wi)
                //print("MicrofacetTransmission wo wi density: ", wo, wi, density)
                return (radiance, wi, density)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                if sameHemisphere(wo, wi) { return 0 }
                let eta = computeEta(cosTheta: cosTheta(wo), eta: self.eta)
                let half = normalized(wo + wi * eta)
                let sqrtDenom = dot(wo, half) + eta * dot(wi, half)
                let dwhDwi = abs(square(eta) * dot(wi, half) / (square(sqrtDenom)))
                //print("probabilityDensity: wo, half, dwhDwi, sqrtDenom: ", wo, half, dwhDwi, sqrtDenom)
                return distribution.pdf(wo: wo, half: half) * dwhDwi
        }

        func albedo() -> Spectrum {
                return t
        }

        var isTransmissive: Bool { return true }

        var t: Spectrum
        var distribution: MicrofacetDistribution
        var eta: (FloatX, FloatX)
        var fresnel: FresnelDielectric
}
