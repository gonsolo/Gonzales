// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

import Foundation  // exit

final class VolumePathIntegrator {

        init(scene: Scene, maxDepth: Int) {
                self.maxDepth = maxDepth
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
                var density: FloatX = 0
                if interaction is SurfaceInteraction {
                        density = bsdf.probabilityDensity(wo: interaction.wo, wi: sample)
                }
                if let mediumInteraction = interaction as? MediumInteraction {
                        density = mediumInteraction.phase.evaluate(wo: mediumInteraction.wo, wi: sample)
                }
                return density
        }

        private func chooseLight(
                sampler: Sampler,
                scene: Scene,
                lightSampler: LightSampler
        ) throws
                -> (Light, FloatX)
        {
                return lightSampler.chooseLight()
        }

        func intersectOrInfiniteLights(
                ray: Ray,
                tHit: inout FloatX,
                bounce: Int,
                l: inout RGBSpectrum,
                interaction: inout SurfaceInteraction,
                scene: Scene,
                hierarchy: Accelerator
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

        private func sampleLightSource(
                light: Light,
                interaction: Interaction,
                sampler: Sampler,
                bsdf: BSDF,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> BSDFSample {
                let zero = BSDFSample()

                let (radiance, wi, lightDensity, visibility) = light.sample(
                        for: interaction, u: sampler.get2D())
                guard !radiance.isBlack && !lightDensity.isInfinite else {
                        return zero
                }
                guard try visibility.unoccluded(hierarchy: hierarchy) else {
                        return zero
                }
                var scatter: RGBSpectrum
                if let mediumInteraction = interaction as? MediumInteraction {
                        let phase = mediumInteraction.phase.evaluate(wo: mediumInteraction.wo, wi: wi)
                        scatter = RGBSpectrum(intensity: phase)
                } else {
                        let reflected = bsdf.evaluate(wo: interaction.wo, wi: wi)
                        let dot = absDot(wi, Vector(normal: interaction.shadingNormal))
                        scatter = reflected * dot
                }
                let estimate = scatter * radiance
                return BSDFSample(estimate, wi, lightDensity)
        }

        private func sampleBrdf(
                light: Light,
                interaction: Interaction,
                sampler: Sampler,
                bsdf: BSDF,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> BSDFSample {

                let zero = BSDFSample()

                var bsdfSample = BSDFSample()
                if let surfaceInteraction = interaction as? SurfaceInteraction {
                        (bsdfSample, _) = try bsdf.sample(wo: surfaceInteraction.wo, u: sampler.get2D())
                        guard bsdfSample.estimate != black && bsdfSample.probabilityDensity > 0 else {
                                return zero
                        }
                        bsdfSample.estimate *= absDot(bsdfSample.incoming, surfaceInteraction.shadingNormal)
                }
                if let mediumInteraction = interaction as? MediumInteraction {
                        let (value, _) = mediumInteraction.phase.samplePhase(
                                wo: interaction.wo,
                                sampler: sampler)
                        bsdfSample.estimate = RGBSpectrum(intensity: value)
                        bsdfSample.probabilityDensity = value
                }
                let ray = interaction.spawnRay(inDirection: bsdfSample.incoming)
                var tHit = FloatX.infinity
                var brdfInteraction = SurfaceInteraction()
                try intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &brdfInteraction,
                        hierarchy: hierarchy)
                if !brdfInteraction.valid {
                        for light in scene.lights {
                                if light is InfiniteLight {
                                        let radiance = light.radianceFromInfinity(for: ray)
                                        bsdfSample.estimate *= radiance
                                        return bsdfSample
                                }
                        }
                        return zero
                }
                return bsdfSample
        }

        private func sampleLight(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> RGBSpectrum {
                let bsdfSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf,
                        scene: scene,
                        hierarchy: hierarchy)
                if bsdfSample.probabilityDensity == 0 {
                        print("light: black")
                        return black
                } else {
                        print("light: ", bsdfSample)
                        return bsdfSample.estimate / bsdfSample.probabilityDensity
                }
        }

        private func sampleBSDF(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> RGBSpectrum {
                let bsdfSample = try sampleBrdf(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf,
                        scene: scene,
                        hierarchy: hierarchy)
                if bsdfSample.probabilityDensity == 0 {
                        return black
                } else {
                        return bsdfSample.estimate / bsdfSample.probabilityDensity
                }
        }

        private func sampleMultipleImportance(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> RGBSpectrum {
                let lightSampler = MultipleImportanceSampler.MISSampler(
                        sample: sampleLightSource, density: lightDensity)
                let brdfSampler = MultipleImportanceSampler.MISSampler(
                        sample: sampleBrdf, density: brdfDensity)
                let misSampler = MultipleImportanceSampler(samplers: (lightSampler, brdfSampler))
                return try misSampler.evaluate(
                        scene: scene,
                        hierarchy: hierarchy,
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
        }

        private func estimateDirect(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator
        ) throws -> RGBSpectrum {
                if light.isDelta {
                        return try sampleLight(
                                light: light,
                                interaction: interaction,
                                bsdf: bsdf,
                                sampler: sampler,
                                scene: scene,
                                hierarchy: hierarchy)
                }
                //return try sampleLight(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler,
                //        scene: scene,
                //        hierarchy: hierarchy)
                //return try sampleBSDF(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler,
                //        scene: scene,
                //        hierarchy: hierarchy)
                return try sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        bsdf: bsdf,
                        sampler: sampler,
                        scene: scene,
                        hierarchy: hierarchy)
        }

        private func sampleOneLight(
                at interaction: Interaction,
                bsdf: BSDF,
                with sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator,
                lightSampler: LightSampler
        ) throws -> RGBSpectrum {
                guard scene.lights.count > 0 else { return black }
                let (light, lightPdf) = try chooseLight(
                        sampler: sampler,
                        scene: scene,
                        lightSampler: lightSampler)
                let estimate = try estimateDirect(
                        light: light,
                        interaction: interaction,
                        bsdf: bsdf,
                        sampler: sampler,
                        scene: scene,
                        hierarchy: hierarchy)
                return estimate / lightPdf
        }

        func russianRoulette(beta: inout RGBSpectrum) -> Bool {
                let roulette = FloatX.random(in: 0..<1)
                let probability: FloatX = 0.5
                if roulette < probability {
                        return true
                } else {
                        beta /= probability
                        return false
                }
        }

        private func sampleMedium(
                beta: RGBSpectrum,
                mediumInteraction: MediumInteraction,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator,
                lightSampler: LightSampler,
                ray: Ray
        ) throws -> (RGBSpectrum, Ray) {
                var ray = ray
                let dummy = BSDF()
                let l =
                        try beta
                        * sampleOneLight(
                                at: mediumInteraction,
                                bsdf: dummy,
                                with: sampler,
                                scene: scene,
                                hierarchy: hierarchy,
                                lightSampler: lightSampler)
                let (_, wi) = mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: sampler)
                ray = mediumInteraction.spawnRay(inDirection: wi)
                return (l, ray)
        }

        private func sampleSurface(
                bounce: Int,
                surfaceInteraction: SurfaceInteraction,
                beta: inout RGBSpectrum,
                ray: Ray,
                albedo: inout RGBSpectrum,
                firstNormal: inout Normal,
                sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator,
                lightSampler: LightSampler
        ) throws -> (RGBSpectrum, Ray, shouldBreak: Bool, shouldContinue: Bool, shouldReturn: Bool) {
                var ray = ray
                var l = black
                if bounce == 0 {
                        if let areaLight = surfaceInteraction.areaLight {
                                l +=
                                        beta
                                        * areaLight.emittedRadiance(
                                                from: surfaceInteraction,
                                                inDirection: surfaceInteraction.wo)
                        }
                }
                guard bounce < maxDepth else {
                        return (black, ray, true, false, false)
                }
                if surfaceInteraction.material == -1 {
                        return (black, ray, true, false, false)
                }
                guard let material = materials[surfaceInteraction.material] else {
                        return (black, ray, true, false, false)
                }
                if material is Interface {
                        ray = surfaceInteraction.spawnRay(inDirection: ray.direction)
                        if let interface = surfaceInteraction.mediumInterface {
                                ray.medium = state.namedMedia[interface.interior]
                        }
                        return (black, ray, false, true, false)
                }
                let bsdf = material.getBSDF(interaction: surfaceInteraction)
                if bounce == 0 {
                        albedo = bsdf.albedo()
                        firstNormal = surfaceInteraction.normal
                }
                let ld =
                        try beta
                        * sampleOneLight(
                                at: surfaceInteraction,
                                bsdf: bsdf,
                                with: sampler,
                                scene: scene,
                                hierarchy: hierarchy,
                                lightSampler: lightSampler)
                l += ld
                let (bsdfSample, _) = try bsdf.sample(
                        wo: surfaceInteraction.wo, u: sampler.get2D())
                guard bsdfSample.probabilityDensity != 0 && !bsdfSample.probabilityDensity.isNaN else {
                        return (l, ray, false, false, true)
                }
                beta = beta * bsdfSample.estimate * absDot(bsdfSample.incoming, surfaceInteraction.normal)
                        / bsdfSample.probabilityDensity
                ray = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                return (l, ray, false, false, false)
        }

        func getRadianceAndAlbedo(
                from ray: Ray,
                tHit: inout FloatX,
                with sampler: Sampler,
                scene: Scene,
                hierarchy: Accelerator,
                lightSampler: LightSampler
        ) throws
                -> (radiance: RGBSpectrum, albedo: RGBSpectrum, normal: Normal)
        {
                var l = black
                var beta = white
                var ray = ray
                var albedo = black
                var firstNormal = Normal()
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
                                hierarchy: hierarchy)
                        if !interaction.valid {
                                break
                        }
                        var mediumL: RGBSpectrum
                        var mediumInteraction: MediumInteraction? = nil
                        if let medium = ray.medium {
                                (mediumL, mediumInteraction) = medium.sample(
                                        ray: ray,
                                        tHit: tHit,
                                        sampler: sampler)
                                beta *= mediumL
                        }
                        if beta.isBlack {
                                break
                        }
                        if let mediumInteraction {
                                guard bounce < maxDepth else {
                                        break
                                }
                                var mediumRadiance = black
                                (mediumRadiance, ray) = try sampleMedium(
                                        beta: beta,
                                        mediumInteraction: mediumInteraction,
                                        sampler: sampler,
                                        scene: scene,
                                        hierarchy: hierarchy,
                                        lightSampler: lightSampler,
                                        ray: ray)
                                l += mediumRadiance
                        } else {
                                var surfaceRadiance = black
                                var shouldBreak = false
                                var shouldContinue = false
                                var shouldReturn = false
                                (surfaceRadiance, ray, shouldBreak, shouldContinue, shouldReturn) =
                                        try sampleSurface(
                                                bounce: bounce,
                                                surfaceInteraction: interaction,
                                                beta: &beta,
                                                ray: ray,
                                                albedo: &albedo,
                                                firstNormal: &firstNormal,
                                                sampler: sampler,
                                                scene: scene,
                                                hierarchy: hierarchy,
                                                lightSampler: lightSampler)
                                if shouldReturn {
                                        l += surfaceRadiance
                                        return (l, white, Normal())
                                }
                                if shouldBreak {
                                        break
                                }
                                if shouldContinue {
                                        continue
                                }
                                l += surfaceRadiance
                        }
                        tHit = FloatX.infinity
                        if bounce > 3 && russianRoulette(beta: &beta) {
                                break
                        }
                }
                intelHack(&albedo)
                return (radiance: l, albedo, firstNormal)
        }

        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout RGBSpectrum) {
                if albedo.r == albedo.g && albedo.r == albedo.b {
                        albedo.r += 0.01
                }
        }

        var maxDepth: Int
}
