// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

struct PathIntegrator: Integrator {

        func getRadianceAndAlbedo(
                from ray: Ray, tHit: inout FloatX, for scene: Scene, with sampler: Sampler
        ) throws
                -> (radiance: Spectrum, albedo: Spectrum, normal: Normal)
        {
                var l = black
                var beta = white
                var ray = ray
                var albedo = black
                var normal = Normal()
                for bounce in 0...maxDepth {
                        let interaction = try intersectOrInfiniteLights(
                                ray: ray, tHit: &tHit, bounce: bounce, l: &l)
                        if !interaction.valid {
                                break
                        }
                        guard let primitive = interaction.primitive else {
                                break
                        }
                        if bounce == 0 {
                                if let areaLight = primitive as? AreaLight {
                                        l +=
                                                beta
                                                * areaLight.emittedRadiance(
                                                        from: interaction,
                                                        inDirection: interaction.wo)
                                }
                        }
                        guard bounce < maxDepth else {
                                break
                        }
                        //guard let material = primitive as? Material else {
                        //guard let material = interaction.material else {
                        //        break
                        //}
                        if interaction.material == -1 {
                                break
                        }
                        guard let material = materials[interaction.material] else {
                                break
                        }
                        let (bsdf, _) = material.computeScatteringFunctions(
                                interaction: interaction)
                        if bounce == 0 {
                                albedo = bsdf.albedo()
                                normal = interaction.normal
                        }
                        let ld =
                                try beta
                                * sampleOneLight(
                                        at: interaction, bsdf: bsdf, with: sampler, in: scene)
                        l += ld
                        let (f, wi, pdf, _) = try bsdf.sample(
                                wo: interaction.wo, u: sampler.get2D())
                        guard pdf != 0 && !pdf.isNaN else {
                                return (l, white, Normal())
                        }
                        beta = beta * f * absDot(wi, interaction.normal) / pdf
                        ray = interaction.spawnRay(inDirection: wi)
                        tHit = FloatX.infinity
                        if bounce > 3 && russianRoulette(beta: &beta) {
                                break
                        }
                }
                intelHack(&albedo)
                return (radiance: l, albedo, normal)
        }

        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout Spectrum) {
                if albedo.r == albedo.g && albedo.r == albedo.b {
                        albedo.r += 0.01
                }
        }

        var maxDepth: Int
}
