struct DielectricBsdf: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                unimplemented()
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
        let eta: FloatX
}
