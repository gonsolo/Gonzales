import Foundation  // exp

struct CoatedDiffuseBsdf: BxDF {

        init(reflectance: RGBSpectrum, roughness: (FloatX, FloatX)) {
                self.reflectance = reflectance
                self.roughness = roughness
                self.topBxdf = DielectricBsdf(
                        distribution: TrowbridgeReitzDistribution(alpha: (1, 1)),
                        refractiveIndex: 1)
                self.bottomBxdf = DiffuseBsdf(reflectance: reflectance)
        }

        private func evaluateNextEvent(
                depth: Int,
                pathThroughputWeight: inout RGBSpectrum,
                sampler: Sampler,
                z: inout FloatX,
                w: inout Vector,
                exitZ: FloatX,
                wis: BSDFSample,
                wi: Vector
        ) -> RGBSpectrum? {
                var estimate = black
                if depth > 3 && pathThroughputWeight.maxValue < 0.25 {
                        let q = max(0, 1 - pathThroughputWeight.maxValue)
                        if sampler.get1D() < q {
                                return nil
                        }
                        pathThroughputWeight /= 1 - q
                }
                // medium scattering albedo is assumed to be zero
                //let mediumScatteringAlbedo = 0
                //if mediumScatteringAlbedo == 0 {
                if z == thickness {
                        z = 0
                } else {
                        z = thickness
                }
                pathThroughputWeight *= transmittance(dz: thickness, w: w)
                //} else {
                //        unimplemented()
                //}
                if z == exitZ {
                        let bsdfSample = topBxdf.sample(wo: -w, u: sampler.get3D())
                        if !bsdfSample.isValid {
                                return nil
                        }
                        pathThroughputWeight *= bsdfSample.throughputWeight()
                        w = bsdfSample.incoming
                } else {
                        // non-exit interface is diffuse
                        var wt: FloatX = 1
                        if !topBxdf.isSpecular {
                                wt = powerHeuristic(
                                        f: wis.probabilityDensity,
                                        g: bottomBxdf.probabilityDensity(
                                                wo: -w,
                                                wi: -wis.incoming))
                        }
                        let floatWeight =
                                pathThroughputWeight
                                * absCosTheta(wis.incoming)
                                * wt
                                * transmittance(dz: thickness, w: wis.incoming)
                                * wis.estimate
                                / wis.probabilityDensity
                        let eval = bottomBxdf.evaluate(wo: -w, wi: -wis.incoming)
                        estimate += floatWeight * eval
                        let bs = bottomBxdf.sample(wo: -w, u: sampler.get3D())
                        if !bs.isValid {
                                return nil
                        }
                        pathThroughputWeight *= bs.throughputWeight()
                        w = bs.incoming
                        if !topBxdf.isSpecular {
                                let fExit = topBxdf.evaluate(wo: -w, wi: wi)
                                if !fExit.isBlack {
                                        var wt: FloatX = 1
                                        // bottom is always black
                                        let exitPDF = topBxdf.probabilityDensity(
                                                wo: -w,
                                                wi: wi)
                                        wt = powerHeuristic(
                                                f: bs.probabilityDensity,
                                                g: exitPDF)
                                        estimate +=
                                                pathThroughputWeight
                                                * transmittance(dz: thickness, w: bs.incoming)
                                                * fExit * wt
                                }
                        }
                }
                return estimate
        }

        private func evaluateOneSample(
                sampler: Sampler,
                wo: Vector,
                wi: Vector,
                exitZ: FloatX,
                estimate: inout RGBSpectrum
        ) {
                let wos = topBxdf.sample(wo: wo, u: sampler.get3D())
                if !wos.isValid {
                        return
                }
                let u2 = (sampler.get1D(), sampler.get1D(), sampler.get1D())
                let wis = topBxdf.sample(wo: wi, u: u2)
                if !wis.isValid {
                        return
                }
                var pathThroughputWeight = wos.throughputWeight()
                var z = thickness
                var w = wos.incoming
                for depth in 0..<maxDepth {
                        guard
                                let nextEstimate = evaluateNextEvent(
                                        depth: depth,
                                        pathThroughputWeight: &pathThroughputWeight,
                                        sampler: sampler,
                                        z: &z,
                                        w: &w,
                                        exitZ: exitZ,
                                        wis: wis,
                                        wi: wi
                                )
                        else {
                                break
                        }
                        estimate += nextEstimate
                }

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
                let exitZ = thickness
                let numberOfSamples = 1

                var estimate = FloatX(numberOfSamples) * topBxdf.evaluate(wo: wo, wi: wi)
                let sampler = RandomSampler()
                for _ in 0..<numberOfSamples {
                        evaluateOneSample(
                                sampler: sampler,
                                wo: wo,
                                wi: wi,
                                exitZ: exitZ,
                                estimate: &estimate)
                }
                estimate /= FloatX(numberOfSamples)
                return estimate
        }

        func sample(wo: Vector, u: ThreeRandomVariables) -> BSDFSample {
                let bs = topBxdf.sample(wo: wo, u: u)
                if !bs.isValid {
                        return invalidBSDFSample
                }
                if bs.isReflection(wo: wo) {
                        return bs
                }
                var w = bs.incoming
                var estimate = bs.estimate * absCosTheta(bs.incoming)
                var probabilityDensity = bs.probabilityDensity
                var z = thickness

                for depth in 0..<maxDepth {
                        let rrBeta = estimate.maxValue / probabilityDensity
                        if depth > 3 && rrBeta < 0.25 {
                                let q = max(0, 1 - rrBeta)
                                if u.0 < q {
                                        return invalidBSDFSample
                                }
                                probabilityDensity *= 1 - q
                        }
                        if w.z == 0 {
                                return invalidBSDFSample
                        }
                        // albedo is zero
                        z = (z == thickness) ? 0 : thickness
                        estimate *= transmittance(dz: thickness, w: w)
                        let interface: BxDF = (z == 0) ? bottomBxdf : topBxdf
                        // TODO: u.0 is used above too, have to generate a new one
                        let bs = interface.sample(wo: -w, u: u)
                        if !bs.isValid {
                                return invalidBSDFSample
                        }
                        estimate *= bs.estimate
                        probabilityDensity *= bs.probabilityDensity
                        w = bs.incoming
                        if bs.isTransmission(wo: -w) {
                                return BSDFSample(estimate, w, probabilityDensity)
                        }
                        estimate *= absCosTheta(bs.incoming)
                }
                return invalidBSDFSample
        }

        //func probabilityDensity(wo: Vector, wi: Vector) -> FloatX {
        //        unimplemented()
        //}

        private func transmittance(dz: FloatX, w: Vector) -> FloatX {
                if abs(dz) < FloatX.leastNormalMagnitude {
                        return 1
                } else {
                        return exp(-abs(dz / w.z))
                }
        }

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
