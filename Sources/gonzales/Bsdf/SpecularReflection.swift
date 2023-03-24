struct SpecularReflection: BxDF {

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum { return black }

        func sample(wo: Vector, u: Point2F) -> (RGBSpectrum, Vector, FloatX) {
                let wi = Vector(x: -wo.x, y: -wo.y, z: wo.z)
                let radiance =
                        fresnel.evaluate(cosTheta: cosTheta(wi)) * reflectance / absCosTheta(wi)
                let density: FloatX = 1.0
                return (radiance, wi, density)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX { return 0 }

        func albedo() -> RGBSpectrum { return reflectance }

        let reflectance: RGBSpectrum
        let fresnel: Fresnel
}
