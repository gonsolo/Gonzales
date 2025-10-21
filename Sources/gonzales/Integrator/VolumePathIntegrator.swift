// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

final class VolumePathIntegrator: Sendable {

        init(scene: Scene, maxDepth: Int) {
                self.scene = scene
                self.maxDepth = maxDepth
        }

        private func brdfDensity<D: DistributionModel>(
                light: Light,
                wo: Vector,
                distributionModel: D,
                sample: Vector
        ) -> FloatX {
                return distributionModel.evaluateProbabilityDensity(wo: wo, wi: sample)
        }

        private func chooseLight(
                sampler: RandomSampler,
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
                estimate: inout RgbSpectrum,
                interaction: inout SurfaceInteraction
        ) throws {
                try scene.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                let radiance = scene.infiniteLights.reduce(
                        black,
                        { accumulated, light in accumulated + light.radianceFromInfinity(for: ray)
                        }
                )
                if bounce == 0 { estimate += radiance }
        }

        private func sampleLightSource<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> BsdfSample {
                let (radiance, wi, lightDensity, visibility) = light.sample(
                        point: interaction.position, u: sampler.get2D())
                guard !radiance.isBlack && !lightDensity.isInfinite else {
                        return invalidBsdfSample
                }
                guard try !visibility.occluded(scene: scene) else {
                        return invalidBsdfSample
                }
                let scatter = distributionModel.evaluateDistributionFunction(
                        wo: interaction.wo, wi: wi, normal: interaction.shadingNormal)
                let estimate = scatter * radiance
                return BsdfSample(estimate, wi, lightDensity)
        }

        private func sampleDistributionFunction<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> BsdfSample {

                let zero = BsdfSample()
                var bsdfSample = distributionModel.sampleDistributionFunction(
                        wo: interaction.wo, normal: interaction.shadingNormal, sampler: sampler)
                guard bsdfSample.estimate != black && bsdfSample.probabilityDensity > 0 else {
                        return zero
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
                        switch light {
                        case .infinite(let infiniteLight):
                                let radiance = infiniteLight.radianceFromInfinity(for: ray)
                                bsdfSample.estimate *= radiance
                                return bsdfSample  // TODO: Just one infinite light?
                        default:
                                break
                        }
                }
                return zero
        }

        private func sampleLight<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> RgbSpectrum {
                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
                if lightSample.probabilityDensity == 0 {
                        return black
                } else {
                        return lightSample.estimate / lightSample.probabilityDensity
                }
        }

        private func sampleGlobalBsdf<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> RgbSpectrum {
                let bsdfSample = try sampleDistributionFunction(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
                if bsdfSample.probabilityDensity == 0 {
                        return black
                } else {
                        return bsdfSample.estimate / bsdfSample.probabilityDensity
                }
        }

        private func sampleMultipleImportance<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> RgbSpectrum {

                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
                let brdfDensity = brdfDensity(
                        light: light,
                        wo: interaction.wo,
                        distributionModel: distributionModel,
                        sample: lightSample.incoming)
                let lightWeight = powerHeuristic(f: lightSample.probabilityDensity, g: brdfDensity)
                let a =
                        lightSample.probabilityDensity == 0
                        ? black : lightSample.estimate * lightWeight / lightSample.probabilityDensity

                let brdfSample = try sampleDistributionFunction(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
                let lightDensity = try light.probabilityDensityFor(
                        samplingDirection: brdfSample.incoming,
                        from: interaction)
                let brdfWeight = powerHeuristic(f: brdfSample.probabilityDensity, g: lightDensity)
                let b =
                        brdfSample.probabilityDensity == 0
                        ? black : brdfSample.estimate * brdfWeight / brdfSample.probabilityDensity

                return a + b
        }

        private func estimateDirect<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: RandomSampler
        ) throws -> RgbSpectrum {
                if light.isDelta {
                        return try sampleLight(
                                light: light,
                                interaction: interaction,
                                distributionModel: distributionModel,
                                sampler: sampler)
                }
                // For debugging uncomment one of the two following methods:
                //return try sampleLight(
                //        light: light,
                //        interaction: interaction,
                //        sampler: sampler)
                //return try sampleGlobalBsdf(
                //        light: light,
                //        interaction: interaction,
                //        sampler: sampler)

                return try sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
        }

        private func sampleOneLight<I: Interaction, D: DistributionModel>(
                at interaction: I,
                distributionModel: D,
                with sampler: RandomSampler,
                lightSampler: LightSampler
        ) throws -> RgbSpectrum {
                let (light, lightPdf) = try chooseLight(
                        sampler: sampler,
                        lightSampler: lightSampler)
                let estimate = try estimateDirect(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: sampler)
                return estimate / lightPdf
        }

        private func stopWithRussianRoulette(bounce: Int, pathThroughputWeight: inout RgbSpectrum) -> Bool {
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

        private func sampleMedium<D: DistributionModel>(
                pathThroughputWeight: RgbSpectrum,
                mediumInteraction: MediumInteraction,
                distributionModel: D,
                sampler: RandomSampler,
                lightSampler: LightSampler,
                ray: Ray
        ) throws -> (RgbSpectrum, Ray) {
                let estimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: mediumInteraction,
                                distributionModel: distributionModel,
                                with: sampler,
                                lightSampler: lightSampler)
                let (_, wi) = mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: wi)
                return (estimate, spawnedRay)
        }

        func oneBounce(
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                ray: inout Ray,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState
        ) throws -> Bool {
                try intersectOrInfiniteLights(
                        ray: ray,
                        tHit: &tHit,
                        bounce: bounce,
                        estimate: &estimate,
                        interaction: &interaction)
                if !interaction.valid {
                        return false
                } else {
                        let result = try oneBounce2(
                                interaction: &interaction,
                                tHit: &tHit,
                                ray: &ray,
                                bounce: bounce,
                                estimate: &estimate,
                                sampler: sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state
                        )
                        return result
                }
        }

        func mediumEstimate<D: DistributionModel>(
                ray: inout Ray,
                pathThroughputWeight: RgbSpectrum,
                mediumInteraction: MediumInteraction,
                distributionModel: D,
                sampler: RandomSampler,
                lightSampler: LightSampler
        ) throws -> RgbSpectrum {
                var mediumRadiance = black
                (mediumRadiance, ray) = try sampleMedium(
                        pathThroughputWeight: pathThroughputWeight,
                        mediumInteraction: mediumInteraction,
                        distributionModel: distributionModel,
                        sampler: sampler,
                        lightSampler: lightSampler,
                        ray: ray)
                return mediumRadiance
        }

        func surfaceEstimate(
                interaction: inout SurfaceInteraction,
                ray: inout Ray,
                bounce: Int,
                pathThroughputWeight: inout RgbSpectrum,
                estimate: inout RgbSpectrum,
                tHit: inout Float,
                sampler: RandomSampler,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState
        ) throws -> Bool {
                let surfaceInteraction = interaction
                if bounce == 0 {
                        if let areaLight = surfaceInteraction.areaLight {
                                estimate +=
                                        pathThroughputWeight
                                        * areaLight.emittedRadiance(
                                                from: surfaceInteraction,
                                                inDirection: surfaceInteraction.wo)
                        }
                }
                //if surfaceInteraction.material == noMaterial {
                //        return false
                //}
                //let bsdf = surfaceInteraction.getBsdf()
                assert(surfaceInteraction.materialIndex >= 0)
                let bsdf = scene.materials[surfaceInteraction.materialIndex].getBsdf(
                        interaction: surfaceInteraction)

                //if surfaceInteraction.material.isInterface {
                //        //var spawnedRay = surfaceInteraction.spawnRay(
                //        let spawnedRay = surfaceInteraction.spawnRay(
                //                inDirection: ray.direction)
                //        //if let interface = surfaceInteraction.mediumInterface {
                //        //spawnedRay.medium = state.namedMedia[interface.interior]
                //        //}
                //        ray = spawnedRay
                //        try bounces(
                //                ray: &ray,
                //                interaction: &interaction,
                //                tHit: &tHit,
                //                bounce: bounce + 1,
                //                estimate: &estimate,
                //                sampler: sampler,
                //                pathThroughputWeight: &pathThroughputWeight,
                //                lightSampler: lightSampler,
                //                albedo: &albedo,
                //                firstNormal: &firstNormal,
                //                state: state)
                //}
                if bounce == 0 {
                        albedo = bsdf.albedo()
                        firstNormal = surfaceInteraction.normal
                }
                let lightEstimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: surfaceInteraction,
                                distributionModel: bsdf,
                                with: sampler,
                                lightSampler: lightSampler)
                estimate += lightEstimate
                let (bsdfSample, _) = bsdf.sampleWorld(
                        wo: surfaceInteraction.wo, u: sampler.get3D())
                guard
                        bsdfSample.probabilityDensity != 0
                                && !bsdfSample.probabilityDensity.isNaN
                else {
                        return false
                }
                pathThroughputWeight *= bsdfSample.throughputWeight(
                        normal: surfaceInteraction.normal)
                let spawnedRay = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                ray = spawnedRay
                return true
        }

        func oneBounce2(
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                ray: inout Ray,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState
        ) throws -> Bool {
                //let (transmittance, mediumInteraction) =
                //ray.medium?.sample(ray: ray, tHit: tHit, sampler: sampler) ?? (white, nil)
                let (transmittance, mediumInteraction): (RgbSpectrum, MediumInteraction?) = (white, nil)
                pathThroughputWeight *= transmittance
                if pathThroughputWeight.isBlack {
                        return false
                }
                guard bounce < maxDepth else {
                        return false
                }
                if let mediumInteraction {
                        let distributionModel = mediumInteraction.getDistributionModel()
                        estimate += try mediumEstimate(
                                ray: &ray,
                                pathThroughputWeight: pathThroughputWeight,
                                mediumInteraction: mediumInteraction,
                                distributionModel: distributionModel,
                                sampler: sampler,
                                lightSampler: lightSampler)
                } else {
                        let result = try surfaceEstimate(
                                interaction: &interaction,
                                ray: &ray,
                                bounce: bounce,
                                pathThroughputWeight: &pathThroughputWeight,
                                estimate: &estimate,
                                tHit: &tHit,
                                sampler: sampler,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state)
                        guard result else {
                                return false
                        }
                }
                tHit = FloatX.infinity
                if stopWithRussianRoulette(
                        bounce: bounce,
                        pathThroughputWeight: &pathThroughputWeight)
                {
                        return false
                }
                return true
        }

        func bounces(
                ray: inout Ray,
                tHit: inout Float,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState
        ) throws {
                var interaction = SurfaceInteraction()
                for bounce in bounce...maxDepth {
                        let _ = try oneBounce(
                                interaction: &interaction,
                                tHit: &tHit,
                                ray: &ray,
                                bounce: bounce,
                                estimate: &estimate,
                                sampler: sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state
                        )
                }
        }

        func bounces(
                ray: inout Ray,
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState
        ) throws {
                for bounce in bounce...maxDepth {
                        let result = try oneBounce(
                                interaction: &interaction,
                                tHit: &tHit,
                                ray: &ray,
                                bounce: bounce,
                                estimate: &estimate,
                                sampler: sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state
                        )
                        if !result {
                                break
                        }
                }
        }

        func getRadianceAndAlbedo(
                from ray: Ray,
                tHit: inout FloatX,
                with sampler: RandomSampler,
                lightSampler: LightSampler,
                state: ImmutableState
        ) throws
                -> (estimate: RgbSpectrum, albedo: RgbSpectrum, normal: Normal)
        {

                // Path throughput weight
                // The product of all GlobalBsdfs and cosines divided by the pdf
                // Π f |cosθ| / pdf
                var pathThroughputWeight = white

                var estimate = black
                var varRay = ray
                var albedo = black
                var firstNormal = zeroNormal
                try bounces(
                        ray: &varRay,
                        tHit: &tHit,
                        bounce: 0,
                        estimate: &estimate,
                        sampler: sampler,
                        pathThroughputWeight: &pathThroughputWeight,
                        lightSampler: lightSampler,
                        albedo: &albedo,
                        firstNormal: &firstNormal,
                        state: state)

                let estimateAlbedoNormal = (
                        estimate: estimate,
                        albedo: albedo,
                        normal: firstNormal
                )
                return estimateAlbedoNormal
        }

        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout RgbSpectrum) {
                if albedo.r == albedo.g && albedo.r == albedo.b {
                        albedo.r += 0.01
                }
        }

        let scene: Scene
        let maxDepth: Int
}
