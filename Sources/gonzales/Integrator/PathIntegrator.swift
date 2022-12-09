// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

private func sampleBrdf(
        light: Light,
        interaction: Interaction,
        sampler: Sampler,
        bsdf: BSDF,
        scene: Scene
) throws -> (estimate: Spectrum, density: FloatX, sample: Vector) {

        let zero = (black, FloatX(0.0), up)

        var (scatter, wi, bsdfDensity, _) = try bsdf.sample(
                wo: interaction.wo, u: sampler.get2D())
        guard scatter != black && bsdfDensity > 0 else {
                return zero
        }
        scatter *= absDot(wi, interaction.shadingNormal)
        let ray = interaction.spawnRay(inDirection: wi)
        var tHit = FloatX.infinity
        var brdfInteraction = SurfaceInteraction()
        try intersect(
                ray: ray,
                tHit: &tHit,
                interaction: &brdfInteraction,
                hierarchy: scene.primitive)
        if !brdfInteraction.valid {
                for light in scene.lights {
                        if light is InfiniteLight {
                                let radiance = light.radianceFromInfinity(for: ray)
                                let estimate = scatter * radiance
                                return (
                                        estimate: estimate, density: bsdfDensity,
                                        sample: wi
                                )
                        }
                }
                return zero
        }
        let radiance = black
        let estimate = scatter * radiance
        return (estimate: estimate, density: bsdfDensity, sample: wi)
}

private func sampleLightSource(
        light: Light,
        interaction: Interaction,
        sampler: Sampler,
        bsdf: BSDF,
        scene: Scene
) throws -> (
        estimate: Spectrum, density: FloatX, sample: Vector
) {
        let zero = (black, FloatX(0.0), up)

        let (radiance, wi, lightDensity, visibility) = light.sample(
                for: interaction, u: sampler.get2D())
        guard !radiance.isBlack && !lightDensity.isInfinite else {
                return zero
        }
        guard try visibility.unoccluded(scene: scene) else {
                return zero
        }
        let reflected = bsdf.evaluate(wo: interaction.wo, wi: wi)
        let dot = absDot(wi, Vector(normal: interaction.shadingNormal))
        let scatter = reflected * dot
        let estimate = scatter * radiance
        return (estimate: estimate, density: lightDensity, sample: wi)
}

private func lightDensity(
        light: Light,
        interaction: Interaction,
        sample: Vector,
        bsdf: BSDF
) throws -> FloatX {
        return try light.probabilityDensityFor(
                samplingDirection: sample, from: interaction)
}

private func brdfDensity(
        light: Light,
        interaction: Interaction,
        sample: Vector,
        bsdf: BSDF
) -> FloatX {
        let density = bsdf.probabilityDensity(wo: interaction.wo, wi: sample)
        return density
}

private func chooseLight(
        withSampler sampler: Sampler,
        scene: Scene
) throws
        -> (Light, FloatX)
{

        assert(scene.lights.count > 0)
        //guard scene.lights.count + scene.infiniteLights.count > 0 else {
        //        throw RenderError.noLights
        //}
        let u = sampler.get1D()
        let lightNum = Int(u * FloatX(scene.lights.count))
        let light = scene.lights[lightNum]
        let probabilityDensity: FloatX = 1.0 / FloatX(scene.lights.count)
        return (light, probabilityDensity)
}

func intersectOrInfiniteLights(
        ray: Ray,
        tHit: inout FloatX,
        bounce: Int,
        l: inout Spectrum,
        interaction: inout SurfaceInteraction,
        scene: Scene,
        hierarchy: BoundingHierarchy
) throws {
        try intersect(ray: ray, tHit: &tHit, interaction: &interaction, hierarchy: hierarchy)
        if interaction.valid {
                return
        }
        let radiance = scene.infiniteLights.reduce(
                black,
                { accumulated, light in accumulated + light.radianceFromInfinity(for: ray) }
        )
        if bounce == 0 { l += radiance }
}

private func sampleOneLight(
        at interaction: SurfaceInteraction,
        bsdf: BSDF,
        with sampler: Sampler,
        scene: Scene
) throws -> Spectrum {

        guard scene.lights.count > 0 else { return black }
        let (light, lightPdf) = try chooseLight(withSampler: sampler, scene: scene)
        let estimate = try estimateDirect(
                light: light,
                atInteraction: interaction,
                bsdf: bsdf,
                withSampler: sampler,
                scene: scene)
        return estimate / lightPdf
}

private func estimateDirect(
        light: Light,
        atInteraction interaction: SurfaceInteraction,
        bsdf: BSDF,
        withSampler sampler: Sampler,
        scene: Scene
) throws -> Spectrum {

        if light.isDelta {
                let (estimate, density, _) = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf,
                        scene: scene)
                if density == 0 {
                        return black
                } else {
                        return estimate / density
                }
        }

        // Light source sampling only
        //let (estimate, density, _) = try sampleLightSource()
        //if density == 0 {
        //        print("light: black")
        //        return black
        //} else {
        //        print("light: ", estimate / density, estimate, density)
        //        return estimate / density
        //}

        // BRDF sampling only
        //let (estimate, density, _) = try sampleBrdf()
        //print("Brdf: ", estimate, density)
        //if density == 0 {
        //        return black
        //} else {
        //        return estimate / density
        //}

        // Light and BRDF sampling with multiple importance sampling
        let lightSampler = MultipleImportanceSampler<Vector>.MISSampler(
                sample: sampleLightSource, density: lightDensity)
        let brdfSampler = MultipleImportanceSampler<Vector>.MISSampler(
                sample: sampleBrdf, density: brdfDensity)
        let sampler = MultipleImportanceSampler(
                samplers: (lightSampler, brdfSampler),
                light: light,
                interaction: interaction,
                sampler: sampler,
                bsdf: bsdf)
        return try sampler.evaluate(scene: scene)
}

final class PathIntegrator {

        init(scene: Scene, maxDepth: Int) {
                self.scene = scene
                self.maxDepth = maxDepth
        }

        func russianRoulette(beta: inout Spectrum) -> Bool {
                let roulette = FloatX.random(in: 0..<1)
                let probability: FloatX = 0.5
                if roulette < probability {
                        return true
                } else {
                        beta /= probability
                        return false
                }
        }

        @_semantics("optremark")
        func getRadianceAndAlbedo(
                from ray: Ray, tHit: inout FloatX, with sampler: Sampler
        ) throws
                -> (radiance: Spectrum, albedo: Spectrum, normal: Normal)
        {
                var l = black
                var beta = white
                var ray = ray
                var albedo = black
                var normal = Normal()
                var interaction = SurfaceInteraction()
                for bounce in 0...maxDepth {
                        interaction.valid = false
                        try intersectOrInfiniteLights(
                                ray: ray,
                                tHit: &tHit,
                                bounce: bounce,
                                l: &l,
                                interaction: &interaction,
                                scene: scene,
                                hierarchy: scene.primitive)
                        if !interaction.valid {
                                break
                        }
                        if bounce == 0 {
                                //if let areaLight = interaction.areaLight {
                                //        l +=
                                //                beta
                                //                * areaLight.emittedRadiance(
                                //                        from: interaction,
                                //                        inDirection: interaction.wo)
                                //}
                        }
                        guard bounce < maxDepth else {
                                break
                        }
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
                                        at: interaction, bsdf: bsdf, with: sampler, scene: scene)
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

        var scene: Scene
        var maxDepth: Int
}
