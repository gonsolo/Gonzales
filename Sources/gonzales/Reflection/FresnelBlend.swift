final class FresnelBlend: BxDF {

        init(
                diffuseReflection: Spectrum,
                glossyReflection: Spectrum,
                distribution: MicrofacetDistribution
        ) {
                self.diffuseReflection = diffuseReflection
                self.glossyReflection = glossyReflection
                self.distribution = distribution
        }

        private func pow5(_ x: FloatX) -> FloatX {
                return x * x * x * x * x
        }

        private func schlickFresnel(_ cosTheta: FloatX) -> Spectrum {
                return glossyReflection + pow5(1 - cosTheta) * (white - glossyReflection)
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum {

                func weight(_ vector: Vector) -> FloatX {
                        return 1.0 - pow5(1.0 - 0.5 * absCosTheta(vector))
                }

                func diffuse() -> Spectrum {
                        let constant = 28.0 / (23.0 * FloatX.pi)
                        let result =
                                constant * diffuseReflection * (white - glossyReflection)
                                * weight(wi) * weight(wo)
                        return result
                }

                func specular() -> Spectrum {
                        let wh = normalized(wi + wo)
                        if wh.isZero { return black }
                        let dist = distribution.differentialArea(withNormal: wh)
                        let fresnel = schlickFresnel(dot(wi, wh))
                        let geo = 4 * absDot(wi, wh) * max(absCosTheta(wi), absCosTheta(wo))
                        let result = dist * fresnel / geo
                        return result
                }

                let diffuseScattered = diffuse()
                let specularScattered = specular()
                let scattered = diffuseScattered + specularScattered
                return scattered
        }

        func albedo() -> Spectrum {
                return evaluate(wo: Vector(x: 0, y: 0, z: 1), wi: Vector(x: 0, y: 0, z: 1))
        }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX) {

                func diffuse(u: Point2F) -> Vector {
                        var u = u
                        u[0] = min(2 * u[0], oneMinusEpsilon)
                        var wi = cosineSampleHemisphere(u: u)
                        if wo.z < 0 { wi.z *= -1 }
                        return wi
                }

                func specular(u: Point2F) -> Vector {
                        var u = u
                        u[0] = min(2 * (u[0] - 0.5), oneMinusEpsilon)
                        let half = distribution.sampleHalfVector(wo: wo, u: u)
                        return reflect(vector: wo, by: half)
                }

                func sampleWi(u: Point2F, either: (Point2F) -> Vector, or: (Point2F) -> Vector)
                        -> Vector
                {
                        if u[0] < 0.5 {
                                return diffuse(u: u)
                        } else {
                                return specular(u: u)
                        }
                }

                let wi = sampleWi(u: u, either: diffuse, or: specular)
                let radiance = evaluate(wo: wo, wi: wi)
                let density = probabilityDensity(wo: wo, wi: wi)
                return (radiance, wi, density)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {

                var diffuse: FloatX { return absCosTheta(wi) / FloatX.pi }

                var specular: FloatX {
                        let half = normalized(wo + wi)
                        let pdfHalf = distribution.pdf(wo: wo, half: half)
                        return pdfHalf / (4 * dot(wo, half))
                }

                if !sameHemisphere(wo, wi) { return 0 }
                return 0.5 * (diffuse + specular)
        }

        let diffuseReflection: Spectrum
        let glossyReflection: Spectrum
        let distribution: MicrofacetDistribution
}
