// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

struct VolumePathIntegrator {

        let maxDepth: Int
        let accelerator: Accelerator
        let scene: Scene
        var xoshiro = Xoshiro()

        init(maxDepth: Int, accelerator: Accelerator, scene: Scene) {
                self.maxDepth = maxDepth
                self.accelerator = accelerator
                self.scene = scene
        }
}

extension VolumePathIntegrator {

        private func brdfDensity<D: DistributionModel>(
                light _: Light,
                wo: Vector,
                distributionModel: D,
                sample: Vector
        ) -> FloatX {
                return distributionModel.evaluateProbabilityDensity(wo: wo, wi: sample)
        }

        private func chooseLight(
                sampler _: RandomSampler,
                lightSampler: inout LightSampler,
                scene: Scene
        ) throws
                -> (Light, FloatX)
        {
                return lightSampler.chooseLight(scene: scene)
        }

        private func intersectOrInfiniteLights(
                ray: Ray,
                tHit: inout FloatX,
                bounce: Int,
                estimate: inout RgbSpectrum,
                interaction: inout SurfaceInteraction
        ) throws {
                try accelerator.intersect(
                        scene: scene,
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
                sampler: inout RandomSampler,
                scene: Scene
        ) throws -> BsdfSample {
                let lightSample = light.sample(
                        point: interaction.position, samples: sampler.get2D(), accelerator: accelerator,
                        scene: scene)
                guard !lightSample.radiance.isBlack && !lightSample.pdf.isInfinite else {
                        return invalidBsdfSample
                }
                guard try !lightSample.visibility.occluded(scene: scene, accelerator: accelerator) else {
                        return invalidBsdfSample
                }
                let scatter = distributionModel.evaluateDistributionFunction(
                        wo: interaction.wo, wi: lightSample.direction, normal: interaction.shadingNormal)
                let estimate = scatter * lightSample.radiance
                return BsdfSample(estimate, lightSample.direction, lightSample.pdf)
        }

        private func sampleDistributionFunction<I: Interaction, D: DistributionModel>(
                light _: Light,
                interaction: I,
                distributionModel: D,
                sampler: inout RandomSampler
        ) throws -> BsdfSample {

                let zero = BsdfSample()
                var bsdfSample = distributionModel.sampleDistributionFunction(
                        wo: interaction.wo, normal: interaction.shadingNormal, sampler: &sampler)
                guard bsdfSample.estimate != black && bsdfSample.probabilityDensity > 0 else {
                        return zero
                }
                let ray = interaction.spawnRay(inDirection: bsdfSample.incoming)
                var tHit = FloatX.infinity
                var brdfInteraction = SurfaceInteraction()
                try accelerator.intersect(
                        scene: scene,
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
                sampler: inout RandomSampler,
                scene: Scene
        ) throws -> RgbSpectrum {
                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: &sampler,
                        scene: scene)
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
                sampler: inout RandomSampler
        ) throws -> RgbSpectrum {
                let bsdfSample = try sampleDistributionFunction(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: &sampler)
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
                sampler: inout RandomSampler,
                scene: Scene
        ) throws -> RgbSpectrum {

                let lightSample = try sampleLightSource(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: &sampler,
                        scene: scene)
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
                        sampler: &sampler)
                let lightDensity = try light.probabilityDensityFor(
                        scene: scene,
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
                sampler: inout RandomSampler,
                scene: Scene
        ) throws -> RgbSpectrum {
                if light.isDelta {
                        return try sampleLight(
                                light: light,
                                interaction: interaction,
                                distributionModel: distributionModel,
                                sampler: &sampler,
                                scene: scene)
                }
                // For debugging uncomment one of the two following methods:
                // return try sampleLight(
                //        light: light,
                //        interaction: interaction,
                //        sampler: sampler)
                // return try sampleGlobalBsdf(
                //        light: light,
                //        interaction: interaction,
                //        sampler: sampler)

                return try sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: &sampler,
                        scene: scene)
        }

        private func sampleOneLight<I: Interaction, D: DistributionModel>(
                at interaction: I,
                distributionModel: D,
                with sampler: inout RandomSampler,
                lightSampler: inout LightSampler,
                scene: Scene
        ) throws -> RgbSpectrum {
                let (light, lightPdf) = try chooseLight(
                        sampler: sampler,
                        lightSampler: &lightSampler,
                        scene: scene)
                let estimate = try estimateDirect(
                        light: light,
                        interaction: interaction,
                        distributionModel: distributionModel,
                        sampler: &sampler,
                        scene: scene)
                return estimate / lightPdf
        }

        private mutating func stopWithRussianRoulette(bounce: Int, pathThroughputWeight: inout RgbSpectrum)
                -> Bool
        {
                if pathThroughputWeight.maxValue < 1 && bounce > 1 {
                        let probability: FloatX = max(0, 1 - pathThroughputWeight.maxValue)
                        let roulette = FloatX.random(in: 0..<1, using: &xoshiro)
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
                sampler: inout RandomSampler,
                lightSampler: inout LightSampler,
                ray: Ray,
                scene: Scene
        ) throws -> (RgbSpectrum, Ray) {
                let estimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: mediumInteraction,
                                distributionModel: distributionModel,
                                with: &sampler,
                                lightSampler: &lightSampler,
                                scene: scene)
                let (_, wi) = mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: &sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: wi)
                return (estimate, spawnedRay)
        }

        private func mediumEstimate<D: DistributionModel>(
                ray: inout Ray,
                pathThroughputWeight: RgbSpectrum,
                mediumInteraction: MediumInteraction,
                distributionModel: D,
                sampler: inout RandomSampler,
                lightSampler: inout LightSampler,
                scene: Scene
        ) throws -> RgbSpectrum {
                var mediumRadiance = black
                (mediumRadiance, ray) = try sampleMedium(
                        pathThroughputWeight: pathThroughputWeight,
                        mediumInteraction: mediumInteraction,
                        distributionModel: distributionModel,
                        sampler: &sampler,
                        lightSampler: &lightSampler,
                        ray: ray,
                        scene: scene)
                return mediumRadiance
        }

        private func surfaceEstimate(
                interaction: inout SurfaceInteraction,
                ray: inout Ray,
                bounce: Int,
                pathThroughputWeight: inout RgbSpectrum,
                estimate: inout RgbSpectrum,
                tHit _: inout Float,
                sampler: inout RandomSampler,
                lightSampler: inout LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state _: ImmutableState,
                scene: Scene
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
                // if surfaceInteraction.material == noMaterial {
                //        return false
                // }
                // let bsdf = surfaceInteraction.getBsdf()
                assert(surfaceInteraction.materialIndex >= 0)
                let bsdf = scene.materials[surfaceInteraction.materialIndex].getBsdf(
                        interaction: surfaceInteraction)

                // if surfaceInteraction.material.isInterface {
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
                // }
                if bounce == 0 {
                        albedo = bsdf.albedo()
                        firstNormal = surfaceInteraction.normal
                }
                let lightEstimate =
                        try pathThroughputWeight
                        * sampleOneLight(
                                at: surfaceInteraction,
                                distributionModel: bsdf,
                                with: &sampler,
                                lightSampler: &lightSampler,
                                scene: scene)
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

        private mutating func oneBounce(
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                ray: inout Ray,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: inout RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: inout LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState,
                scene: Scene
        ) throws -> Bool {

                try intersectOrInfiniteLights(
                        ray: ray,
                        tHit: &tHit,
                        bounce: bounce,
                        estimate: &estimate,
                        interaction: &interaction)
                if !interaction.valid {
                        return false  // No surface hit, so stop this bounce
                }
                // let (transmittance, mediumInteraction) = ray.medium?.sample(...) ?? (white, nil)
                let (transmittance, mediumInteraction): (RgbSpectrum, MediumInteraction?) = (white, nil)  // Keeping original implementation
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
                                sampler: &sampler,
                                lightSampler: &lightSampler,
                                scene: scene)
                } else {
                        // Surface hit: Perform lighting and generate new ray
                        let result = try surfaceEstimate(
                                interaction: &interaction,
                                ray: &ray,
                                bounce: bounce,
                                pathThroughputWeight: &pathThroughputWeight,
                                estimate: &estimate,
                                tHit: &tHit,
                                sampler: &sampler,
                                lightSampler: &lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state,
                                scene: scene)
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

        private mutating func bounces(
                ray: inout Ray,
                tHit: inout Float,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: inout RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: inout LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState,
                scene: Scene
        ) throws {
                var interaction = SurfaceInteraction()
                for bounce in bounce...maxDepth {
                        _ = try oneBounce(
                                interaction: &interaction,
                                tHit: &tHit,
                                ray: &ray,
                                bounce: bounce,
                                estimate: &estimate,
                                sampler: &sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: &lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state,
                                scene: scene
                        )
                }
        }

        private mutating func bounces(
                ray: inout Ray,
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: inout RandomSampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: inout LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                state: ImmutableState,
                scene: Scene
        ) throws {
                for bounce in bounce...maxDepth {
                        let result = try oneBounce(
                                interaction: &interaction,
                                tHit: &tHit,
                                ray: &ray,
                                bounce: bounce,
                                estimate: &estimate,
                                sampler: &sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: &lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal,
                                state: state,
                                scene: scene)
                        if !result {
                                break
                        }
                }
        }
}

struct RayTraceSample {
        let estimate: RgbSpectrum
        let albedo: RgbSpectrum
        let normal: Normal
}

extension VolumePathIntegrator {

        mutating func evaluateRayPath(
                from ray: Ray,
                tHit: inout FloatX,
                with sampler: inout RandomSampler,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws
                -> RayTraceSample
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
                        sampler: &sampler,
                        pathThroughputWeight: &pathThroughputWeight,
                        lightSampler: &lightSampler,
                        albedo: &albedo,
                        firstNormal: &firstNormal,
                        state: state,
                        scene: scene)

                return RayTraceSample(
                        estimate: estimate,
                        albedo: albedo,
                        normal: firstNormal
                )
        }

        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout RgbSpectrum) {
                if albedo.red == albedo.green && albedo.red == albedo.blue {
                        albedo.red += 0.01
                }
        }

}
