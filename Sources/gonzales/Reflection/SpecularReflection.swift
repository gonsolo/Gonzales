struct SpecularReflection: BxDF {

        init(reflectance: Spectrum, fresnel: Fresnel) {
                self.reflectance = reflectance
                self.fresnel = fresnel
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum { return black }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, FloatX) {
                let wi = Vector(x: -wo.x, y: -wo.y, z: wo.z)
                let radiance =
                        fresnel.evaluate(cosTheta: cosTheta(wi)) * reflectance / absCosTheta(wi)
                let density: FloatX = 1.0
                return (radiance, wi, density)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX { return 0 }

        func albedo() -> Spectrum { return reflectance }

        let reflectance: Spectrum
        let fresnel: Fresnel
}
