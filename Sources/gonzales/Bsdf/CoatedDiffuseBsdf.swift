struct CoatedDiffuseBxdf: BxDF {

        init(reflectance: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.reflectance = reflectance
                self.roughness = roughness
                self.topBxdf = Dielectric()
                self.bottomBxdf = DiffuseBxdf(reflectance: reflectance)
        }

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                // TODO
                return bottomBxdf.evaluate(wo: wo, wi: wi)
        }

        //func sample(wo: Vector, u: Point2F) -> (RGBSpectrum, Vector, FloatX) {
        //        unimplemented()
        //}

        //func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
        //        unimplemented()
        //}

        func albedo() -> RGBSpectrum { return reflectance }

        let reflectance: RGBSpectrum
        let roughness: (FloatX, FloatX)

        let topBxdf: Dielectric
        let bottomBxdf: DiffuseBxdf
}
