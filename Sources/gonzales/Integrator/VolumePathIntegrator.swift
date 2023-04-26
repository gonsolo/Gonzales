// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

import Foundation  // exit

final class VolumePathIntegrator {

        init(scene: Scene, maxDepth: Int) {
                self.scene = scene
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
                lightSampler: LightSampler
        ) throws
                -> (Light, FloatX)
        {
                return lightSampler.chooseLight()
        }

        private func intersectOrInfiniteLights(
                ray: Ray,
                tHit: inout FloatX,
                bounce: Int,
                estimate: inout RGBSpectrum,
                interaction: inout SurfaceInteraction
        ) throws {
                try scene.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                if interaction.valid {
                        return
                }
                let radiance = scene.infiniteLights.reduce(
                        black,
                        { accumulated, light in accumulated + light.radianceFromInfinity(for: ray) }
                )
                if bounce == 0 { estimate += radiance }
        }

        private func sampleLightSource(
                light: Light,
                interaction: Interaction,
                sampler: Sampler,
                bsdf: BSDF
        ) throws -> BSDFSample {
                let zero = BSDFSample()

                let (radiance, wi, lightDensity, visibility) = light.sample(
                        for: interaction, u: sampler.get2D())
                guard !radiance.isBlack && !lightDensity.isInfinite else {
                        return zero
                }
                guard try visibility.unoccluded(scene: scene) else {
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
                bsdf: BSDF
        ) throws -> BSDFSample {

                let zero = BSDFSample()

                var bsdfSample = BSDFSample()
                if let surfaceInteraction = interaction as? SurfaceInteraction {
                        (bsdfSample, _) = try bsdf.sample(wo: surfaceInteraction.wo, u: sampler.get3D())
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
                try scene.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &brdfInteraction)
                if brdfInteraction.valid {
                        return zero
                }
                for light in scene.lights {
                        if light is InfiniteLight {
                                let radiance = light.radianceFromInfinity(for: ray)
                                bsdfSample.estimate *= radiance
                                return bsdfSample  // TODO: Just one infinite light?
                        }
                }
                return zero
        }

        private func sampleLight(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler
        ) throws -> RGBSpectrum {
                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
                if lightSample.probabilityDensity == 0 {
                        print("light: black")
                        return black
                } else {
                        print("light: ", lightSample)
                        return lightSample.estimate / lightSample.probabilityDensity
                }
        }

        private func sampleBSDF(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler
        ) throws -> RGBSpectrum {
                let bsdfSample = try sampleBrdf(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
                if bsdfSample.probabilityDensity == 0 {
                        print("light: black")
                        return black
                } else {
                        print("light: ", bsdfSample)
                        return bsdfSample.estimate / bsdfSample.probabilityDensity
                }
        }

        private func sampleMultipleImportance(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler
        ) throws -> RGBSpectrum {
                let lightSampler = MultipleImportanceSampler.MISSampler(
                        sample: sampleLightSource, density: lightDensity)
                let brdfSampler = MultipleImportanceSampler.MISSampler(
                        sample: sampleBrdf, density: brdfDensity)
                let misSampler = MultipleImportanceSampler(
                        scene: scene,
                        samplers: (lightSampler, brdfSampler))
                return try misSampler.evaluate(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
        }

        private func estimateDirect(
                light: Light,
                interaction: Interaction,
                bsdf: BSDF,
                sampler: Sampler
        ) throws -> RGBSpectrum {
                if light.isDelta {
                        return try sampleLight(
                                light: light,
                                interaction: interaction,
                                bsdf: bsdf,
                                sampler: sampler)
                }
                // For debugging uncomment one of the two following methods:
                //return try sampleLight(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler)
                //return try sampleBSDF(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler)
                return try sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        bsdf: bsdf,
                        sampler: sampler)
        }

        private func sampleOneLight(
                at interaction: Interaction,
                bsdf: BSDF,
                with sampler: Sampler,
                lightSampler: LightSampler
        ) throws -> RGBSpectrum {
                guard scene.lights.count > 0 else { return black }
                let (light, lightPdf) = try chooseLight(
                        sampler: sampler,
                        lightSampler: lightSampler)
                let estimate = try estimateDirect(
                        light: light,
                        interaction: interaction,
                        bsdf: bsdf,
                        sampler: sampler)
                return estimate / lightPdf
        }

        private func stopWithRussianRoulette(bounce: Int, pathThroughputWeight: inout RGBSpectrum) -> Bool {
                if pathThroughputWeight.maxValue < 1 && bounce > 1 {
                        let probability: FloatX = max(0, 1 - pathThroughputWeight.maxValue)
                        let roulette = FloatX.random(in: 0..<1)
                        if roulette < probability {
                                return true
                        }
                        pathThroughputWeight /= 1 - probability
                }
                return false
        }

        private func sampleMedium(
                pathThroughputWeight: RGBSpectrum,
                mediumInteraction: MediumInteraction,
                sampler: Sampler,
                lightSampler: LightSampler,
                ray: Ray
        ) throws -> (RGBSpectrum, Ray) {
                let dummy = BSDF()
                let estimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: mediumInteraction,
                                bsdf: dummy,
                                with: sampler,
                                lightSampler: lightSampler)
                let (_, wi) = mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: wi)
                return (estimate, spawnedRay)
        }

        private func sampleSurface(
                bounce: Int,
                surfaceInteraction: SurfaceInteraction,
                pathThroughputWeight: inout RGBSpectrum,
                ray: Ray,
                albedo: inout RGBSpectrum,
                firstNormal: inout Normal,
                sampler: Sampler,
                lightSampler: LightSampler
        ) throws -> (RGBSpectrum, Ray, shouldBreak: Bool, shouldContinue: Bool, shouldReturn: Bool) {
                var estimate = black
                if bounce == 0 {
                        if let areaLight = surfaceInteraction.areaLight {
                                estimate +=
                                        pathThroughputWeight
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
                        var spawnedRay = surfaceInteraction.spawnRay(inDirection: ray.direction)
                        if let interface = surfaceInteraction.mediumInterface {
                                spawnedRay.medium = state.namedMedia[interface.interior]
                        }
                        return (estimate, spawnedRay, false, true, false)
                }
                let bsdf = material.getBSDF(interaction: surfaceInteraction)
                if bounce == 0 {
                        albedo = bsdf.albedo()
                        firstNormal = surfaceInteraction.normal
                }
                let lightEstimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: surfaceInteraction,
                                bsdf: bsdf,
                                with: sampler,
                                lightSampler: lightSampler)
                estimate += lightEstimate
                let (bsdfSample, _) = try bsdf.sample(
                        wo: surfaceInteraction.wo, u: sampler.get3D())
                guard bsdfSample.probabilityDensity != 0 && !bsdfSample.probabilityDensity.isNaN else {
                        return (estimate, ray, false, false, true)
                }
                pathThroughputWeight *= bsdfSample.throughputWeight(normal: surfaceInteraction.normal)
                let spawnedRay = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                return (estimate, spawnedRay, false, false, false)
        }

        func getRadianceAndAlbedo(
                from ray: Ray,
                tHit: inout FloatX,
                with sampler: Sampler,
                lightSampler: LightSampler
        ) throws
                -> (estimate: RGBSpectrum, albedo: RGBSpectrum, normal: Normal)
        {
                var estimate = black

                // Path throughput weight
                // The product of all BSDFs and cosines divided by the pdf
                // Π f |cosθ| / pdf
                var pathThroughputWeight = white

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
                                estimate: &estimate,
                                interaction: &interaction)
                        if !interaction.valid {
                                break
                        }
                        let (mediumL, mediumInteraction) =
                                ray.medium?.sample(ray: ray, tHit: tHit, sampler: sampler) ?? (white, nil)
                        pathThroughputWeight *= mediumL
                        if pathThroughputWeight.isBlack {
                                break
                        }
                        if let mediumInteraction {
                                guard bounce < maxDepth else {
                                        break
                                }
                                var mediumRadiance = black
                                (mediumRadiance, ray) = try sampleMedium(
                                        pathThroughputWeight: pathThroughputWeight,
                                        mediumInteraction: mediumInteraction,
                                        sampler: sampler,
                                        lightSampler: lightSampler,
                                        ray: ray)
                                estimate += mediumRadiance
                        } else {
                                var surfaceRadiance = black
                                var shouldBreak = false
                                var shouldContinue = false
                                var shouldReturn = false
                                (surfaceRadiance, ray, shouldBreak, shouldContinue, shouldReturn) =
                                        try sampleSurface(
                                                bounce: bounce,
                                                surfaceInteraction: interaction,
                                                pathThroughputWeight: &pathThroughputWeight,
                                                ray: ray,
                                                albedo: &albedo,
                                                firstNormal: &firstNormal,
                                                sampler: sampler,
                                                lightSampler: lightSampler)
                                if shouldReturn {
                                        estimate += surfaceRadiance
                                        return (estimate, white, Normal())
                                }
                                if shouldBreak {
                                        break
                                }
                                if shouldContinue {
                                        continue
                                }
                                estimate += surfaceRadiance
                        }
                        tHit = FloatX.infinity
                        if stopWithRussianRoulette(
                                bounce: bounce,
                                pathThroughputWeight: &pathThroughputWeight)
                        {
                                break
                        }
                }
                intelHack(&albedo)
                return (estimate: estimate, albedo: albedo, normal: firstNormal)
        }

        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout RGBSpectrum) {
                if albedo.r == albedo.g && albedo.r == albedo.b {
                        albedo.r += 0.01
                }
        }

        let scene: Scene
        var maxDepth: Int
}
