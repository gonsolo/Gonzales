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
                sample: Vector
        ) throws -> FloatX {
                return try light.probabilityDensityFor(
                        samplingDirection: sample, from: interaction)
        }

        private func brdfDensity(
                light: Light,
                interaction: Interaction,
                sample: Vector
        ) -> FloatX {
                var density: FloatX = 0
                if let surfaceInteraction = interaction as? SurfaceInteraction {
                        density = surfaceInteraction.bsdf.probabilityDensityWorld(
                                wo: interaction.wo,
                                wi: sample)
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
                sampler: Sampler
        ) throws -> BsdfSample {
                let zero = BsdfSample()

                let (radiance, wi, lightDensity, visibility) = light.sample(
                        for: interaction, u: sampler.get2D())
                guard !radiance.isBlack && !lightDensity.isInfinite else {
                        return zero
                }
                guard try visibility.unoccluded(scene: scene) else {
                        return zero
                }
                let scatter = interaction.evaluateDistributionFunction(wi: wi)
                let estimate = scatter * radiance
                return BsdfSample(estimate, wi, lightDensity)
        }

        private func sampleBrdf(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) throws -> BsdfSample {

                let zero = BsdfSample()

                var bsdfSample = BsdfSample()
                if let surfaceInteraction = interaction as? SurfaceInteraction {
                        (bsdfSample, _) = try surfaceInteraction.bsdf.sampleWorld(
                                wo: surfaceInteraction.wo,
                                u: sampler.get3D())
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
                sampler: Sampler
        ) throws -> RGBSpectrum {
                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                if lightSample.probabilityDensity == 0 {
                        print("light: black")
                        return black
                } else {
                        print("light: ", lightSample)
                        return lightSample.estimate / lightSample.probabilityDensity
                }
        }

        private func sampleGlobalBsdf(
                light: Light,
                interaction: Interaction,
                bsdf: GlobalBsdf,
                sampler: Sampler
        ) throws -> RGBSpectrum {
                let bsdfSample = try sampleBrdf(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
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
                sampler: Sampler,
                bsdf: GlobalBsdf
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
                sampler: Sampler
        ) throws -> RGBSpectrum {
                if light.isDelta {
                        return try sampleLight(
                                light: light,
                                interaction: interaction,
                                sampler: sampler)
                }
                // For debugging uncomment one of the two following methods:
                //return try sampleLight(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler)
                //return try sampleGlobalBsdf(
                //        light: light,
                //        interaction: interaction,
                //        bsdf: bsdf,
                //        sampler: sampler)
                var bsdf: GlobalBsdf = DummyBsdf()
                if let surfaceInteraction = interaction as? SurfaceInteraction {
                        bsdf = surfaceInteraction.bsdf
                }

                return try sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
        }

        private func sampleOneLight(
                at interaction: Interaction,
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
                let estimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: mediumInteraction,
                                with: sampler,
                                lightSampler: lightSampler)
                let (_, wi) = mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: wi)
                return (estimate, spawnedRay)
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
                // The product of all GlobalBsdfs and cosines divided by the pdf
                // Π f |cosθ| / pdf
                var pathThroughputWeight = white

                var ray = ray
                var albedo = black
                var firstNormal = zeroNormal
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
                        guard bounce < maxDepth else {
                                break
                        }
                        if let mediumInteraction {
                                var mediumRadiance = black
                                (mediumRadiance, ray) = try sampleMedium(
                                        pathThroughputWeight: pathThroughputWeight,
                                        mediumInteraction: mediumInteraction,
                                        sampler: sampler,
                                        lightSampler: lightSampler,
                                        ray: ray)
                                estimate += mediumRadiance
                        } else {
                                var surfaceInteraction = interaction
                                if bounce == 0 {
                                        if let areaLight = surfaceInteraction.areaLight {
                                                estimate +=
                                                        pathThroughputWeight
                                                        * areaLight.emittedRadiance(
                                                                from: surfaceInteraction,
                                                                inDirection: surfaceInteraction.wo)
                                        }
                                }
                                if surfaceInteraction.material == noMaterial {
                                        break
                                }
                                guard let material = materials[surfaceInteraction.material] else {
                                        break
                                }
                                if material is Interface {
                                        var spawnedRay = surfaceInteraction.spawnRay(
                                                inDirection: ray.direction)
                                        if let interface = surfaceInteraction.mediumInterface {
                                                spawnedRay.medium = state.namedMedia[interface.interior]
                                        }
                                        ray = spawnedRay
                                        continue
                                }
                                material.setBsdf(interaction: &surfaceInteraction)
                                if bounce == 0 {
                                        albedo = surfaceInteraction.bsdf.albedo()
                                        firstNormal = surfaceInteraction.normal
                                }
                                let lightEstimate =
                                        try pathThroughputWeight
                                        * sampleOneLight(
                                                at: surfaceInteraction,
                                                with: sampler,
                                                lightSampler: lightSampler)
                                estimate += lightEstimate
                                let (bsdfSample, _) = try surfaceInteraction.bsdf.sampleWorld(
                                        wo: surfaceInteraction.wo, u: sampler.get3D())
                                guard
                                        bsdfSample.probabilityDensity != 0
                                                && !bsdfSample.probabilityDensity.isNaN
                                else {
                                        return (estimate, albedo, firstNormal)
                                }
                                pathThroughputWeight *= bsdfSample.throughputWeight(
                                        normal: surfaceInteraction.normal)
                                let spawnedRay = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                                ray = spawnedRay
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
