struct SpecularReflection: LocalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum { return black }

        func sampleLocal(wo: Vector, u: Point2F) -> BsdfSample {
                let wi = Vector(x: -wo.x, y: -wo.y, z: wo.z)
                let radiance =
                        fresnel.evaluate(cosTheta: cosTheta(wi)) * reflectance / absCosTheta(wi)
                let density: FloatX = 1.0
                return BsdfSample(radiance, wi, density)
        }

        func probabilityDensityLocal(wo: Vector, wi: Vector) -> FloatX { return 0 }

        func albedoLocal() -> RGBSpectrum { return reflectance }

        let reflectance: RGBSpectrum
        let fresnel: Fresnel
}
