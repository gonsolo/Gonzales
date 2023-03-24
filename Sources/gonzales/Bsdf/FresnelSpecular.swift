struct FresnelSpecular: BxDF {

        init(reflectance: RGBSpectrum, transmittance: RGBSpectrum, etaA: FloatX, etaB: FloatX) {
                self.reflectance = reflectance
                self.transmittance = transmittance
                self.etaA = etaA
                self.etaB = etaB
        }

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum { return black }

        func sample(wo: Vector, u: Point2F) -> (RGBSpectrum, Vector, FloatX) {

                func reflective(fresnel: FloatX) -> (RGBSpectrum, Vector, FloatX) {
                        let wi = Vector(x: -wo.x, y: -wo.y, z: wo.z)
                        let radiance = dielectric * reflectance * absCosTheta(wi)
                        let pdf = dielectric
                        return (radiance, wi, pdf)
                }

                func transmittive(fresnel: FloatX) -> (RGBSpectrum, Vector, FloatX) {
                        var etaI = etaA
                        var etaT = etaB
                        let entering = cosTheta(wo) > 0
                        if !entering {
                                swap(&etaI, &etaT)
                        }
                        let normal = faceforward(normal: Normal(x: 0, y: 0, z: 1), comparedTo: wo)
                        guard let wt = refract(wi: wo, normal: normal, eta: etaI / etaT) else {
                                return (black, nullVector, 0)
                        }
                        let pdf = 1 - dielectric
                        let radiance = transmittance * (1 - dielectric)
                        return (radiance, wt, pdf)
                }

                let dielectric = frDielectric(cosThetaI: cosTheta(wo), etaI: etaA, etaT: etaB)
                if u.x < dielectric {
                        return reflective(fresnel: dielectric)
                } else {
                        return transmittive(fresnel: dielectric)
                }
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
                return 1
        }

        func albedo() -> RGBSpectrum { return white }

        var isReflective: Bool { return true }
        var isTransmissive: Bool { return true }

        var reflectance: RGBSpectrum
        var transmittance: RGBSpectrum
        var etaA: FloatX
        var etaB: FloatX
}
