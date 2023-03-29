struct CoatedDiffuseBsdf: BxDF {

        init(reflectance: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.reflectance = reflectance
                self.roughness = roughness
                self.topBxdf = DielectricBsdf(
                        distribution: TrowbridgeReitzDistribution(alpha: (1, 1)),
                        refractiveIndex: 1)
                self.bottomBxdf = DiffuseBsdf(reflectance: reflectance)
        }

        func evaluate(wo: Vector, wi: Vector) -> RGBSpectrum {
                assert(wo.z > 0)
                assert(sameHemisphere(wi, wo))

                // twoSided always true
                // enteredTop always true
                // enterInterface always top
                // exitInterface always top
                // nonExitInterface always bottom
                // sameHemisphere always true
                // exitZ = thickness
                let numberOfSamples = 1

                let estimate = FloatX(numberOfSamples) * topBxdf.evaluate(wo: wo, wi: wi)
                _ = estimate
                let sampler = RandomSampler()
                for _ in 0..<numberOfSamples {
                        let u1 = (sampler.get1D(), sampler.get1D(), sampler.get1D())
                        let topSample = topBxdf.sample(wo: wo, u: u1)
                        if topSample.estimate.isBlack || topSample.probabilityDensity.isZero
                                || topSample.incoming.z.isZero
                        {
                                continue
                        }
                        let u2 = (sampler.get1D(), sampler.get1D(), sampler.get1D())
                        let bottomSample = bottomBxdf.sample(wo: wi, u: u2)
                        if bottomSample.estimate.isBlack || bottomSample.probabilityDensity.isZero
                                || bottomSample.incoming.z.isZero
                        {
                                continue
                        }
                        let beta =
                                topSample.estimate * absCosTheta(topSample.incoming)
                                / topSample.probabilityDensity
                        let z = thickness
                        let w = topSample.incoming
                        let phase = HenyeyGreenstein()
                        for depth in 0..<maxDepth {

                        }

                }
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

        let thickness: FloatX = 0.1

        let maxDepth = 10
        // g in PBRT
        let asymmetry: FloatX = 0

        let reflectance: RGBSpectrum
        let roughness: (FloatX, FloatX)

        let topBxdf: DielectricBsdf
        let bottomBxdf: DiffuseBsdf
}
