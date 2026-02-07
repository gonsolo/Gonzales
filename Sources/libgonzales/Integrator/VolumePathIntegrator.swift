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

struct RayTraceSample {
        let estimate: RgbSpectrum
        let albedo: RgbSpectrum
        let normal: Normal
}

extension VolumePathIntegrator {

        mutating func evaluateRayPath(
                from ray: Ray,
                tHit: inout FloatX,
                with sampler: inout Sampler,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws -> RayTraceSample {

                // Path throughput weight
                // The product of all GlobalBsdfs and cosines divided by the pdf
                // Π f |cosθ| / pdf
                var bounceState = BounceState(
                        ray: ray,
                        tHit: tHit,
                        bounce: 0,
                        estimate: black,
                        throughput: white,
                        albedo: black,
                        firstNormal: zeroNormal)
                var context = IntegratorContext(
                        sampler: sampler,
                        lightSampler: lightSampler,
                        state: state,
                        scene: scene)

                try bounces(state: &bounceState, context: &context)

                tHit = bounceState.tHit
                sampler = context.sampler
                lightSampler = context.lightSampler

                return RayTraceSample(
                        estimate: bounceState.estimate,
                        albedo: bounceState.albedo,
                        normal: bounceState.firstNormal
                )
        }
}

extension VolumePathIntegrator {
        private struct BounceState {
                var ray: Ray
                var tHit: FloatX
                var bounce: Int
                var estimate: RgbSpectrum
                var throughput: RgbSpectrum
                var albedo: RgbSpectrum
                var firstNormal: Normal
                var interaction: SurfaceInteraction = SurfaceInteraction()
        }

        private struct IntegratorContext {
                var sampler: Sampler
                var lightSampler: LightSampler
                let state: ImmutableState
                let scene: Scene
        }

        private func brdfDensity<D: DistributionModel>(
                light _: Light,
                outgoing: Vector,
                distributionModel: D,
                sample: Vector
        ) -> FloatX {
                return distributionModel.evaluateProbabilityDensity(outgoing: outgoing, incident: sample)
        }

        private func chooseLight(
                sampler _: Sampler,
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
                sampler: inout Sampler,
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
                        outgoing: interaction.outgoing,
                        incident: lightSample.direction,
                        normal: interaction.shadingNormal)
                let estimate = scatter * lightSample.radiance
                return BsdfSample(estimate, lightSample.direction, lightSample.pdf)
        }

        private func sampleDistributionFunction<I: Interaction, D: DistributionModel>(
                light _: Light,
                interaction: I,
                distributionModel: D,
                sampler: inout Sampler
        ) throws -> BsdfSample {

                let zero = BsdfSample()
                var bsdfSample = distributionModel.sampleDistributionFunction(
                        outgoing: interaction.outgoing, normal: interaction.shadingNormal, sampler: &sampler)
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
                                return bsdfSample
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
                sampler: inout Sampler,
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
                sampler: inout Sampler
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
                sampler: inout Sampler,
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
                        outgoing: interaction.outgoing,
                        distributionModel: distributionModel,
                        sample: lightSample.incoming)
                let lightWeight = powerHeuristic(pdfF: lightSample.probabilityDensity, pdfG: brdfDensity)
                let lightContribution =
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
                let brdfWeight = powerHeuristic(pdfF: brdfSample.probabilityDensity, pdfG: lightDensity)
                let brdfContribution =
                        brdfSample.probabilityDensity == 0
                        ? black : brdfSample.estimate * brdfWeight / brdfSample.probabilityDensity

                return lightContribution + brdfContribution
        }

        private func estimateDirect<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: inout Sampler,
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
                with sampler: inout Sampler,
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
                state: BounceState,
                mediumInteraction: MediumInteraction,
                distributionModel: D,
                context: inout IntegratorContext
        ) throws -> (RgbSpectrum, Ray) {
                let estimate =
                        try state.throughput
                        * sampleOneLight(
                                at: mediumInteraction,
                                distributionModel: distributionModel,
                                with: &context.sampler,
                                lightSampler: &context.lightSampler,
                                scene: context.scene)
                let (_, incident) = mediumInteraction.phase.samplePhase(
                        outgoing: -state.ray.direction,
                        sampler: &context.sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: incident)
                return (estimate, spawnedRay)
        }

        private func mediumEstimate<D: DistributionModel>(
                state: inout BounceState,
                mediumInteraction: MediumInteraction,
                distributionModel: D,
                context: inout IntegratorContext
        ) throws -> RgbSpectrum {
                var mediumRadiance = black
                (mediumRadiance, state.ray) = try sampleMedium(
                        state: state,
                        mediumInteraction: mediumInteraction,
                        distributionModel: distributionModel,
                        context: &context)
                return mediumRadiance
        }

        private func surfaceEstimate(
                state: inout BounceState,
                context: inout IntegratorContext
        ) throws -> Bool {
                let surfaceInteraction = state.interaction
                if state.bounce == 0 {
                        if let areaLight = surfaceInteraction.areaLight {
                                state.estimate +=
                                        state.throughput
                                        * areaLight.emittedRadiance(
                                                from: surfaceInteraction,
                                                inDirection: surfaceInteraction.outgoing)
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
                if state.bounce == 0 {
                        state.albedo = bsdf.albedo()
                        state.firstNormal = surfaceInteraction.normal
                }
                let lightEstimate =
                        try state.throughput
                        * sampleOneLight(
                                at: surfaceInteraction,
                                distributionModel: bsdf,
                                with: &context.sampler,
                                lightSampler: &context.lightSampler,
                                scene: context.scene)
                state.estimate += lightEstimate
                let (bsdfSample, _) = bsdf.sampleWorld(
                        outgoing: surfaceInteraction.outgoing, uSample: context.sampler.get3D())
                guard
                        bsdfSample.probabilityDensity != 0
                                && !bsdfSample.probabilityDensity.isNaN
                else {
                        return false
                }
                state.throughput *= bsdfSample.throughputWeight(normal: surfaceInteraction.normal)
                let spawnedRay = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                state.ray = spawnedRay
                return true
        }

        private mutating func oneBounce(
                state: inout BounceState,
                context: inout IntegratorContext
        ) throws -> Bool {

                try intersectOrInfiniteLights(
                        ray: state.ray,
                        tHit: &state.tHit,
                        bounce: state.bounce,
                        estimate: &state.estimate,
                        interaction: &state.interaction)
                if !state.interaction.valid {
                        return false  // No surface hit, so stop this bounce
                }
                // let (transmittance, mediumInteraction) = ray.medium?.sample(...) ?? (white, nil)
                let (transmittance, mediumInteraction): (RgbSpectrum, MediumInteraction?) = (white, nil)
                state.throughput *= transmittance

                if state.throughput.isBlack {
                        return false
                }
                guard state.bounce < maxDepth else {
                        return false
                }
                if let mediumInteraction {
                        let distributionModel = mediumInteraction.getDistributionModel()
                        state.estimate += try mediumEstimate(
                                state: &state,
                                mediumInteraction: mediumInteraction,
                                distributionModel: distributionModel,
                                context: &context)
                } else {
                        // Surface hit: Perform lighting and generate new ray
                        let result = try surfaceEstimate(
                                state: &state,
                                context: &context)
                        guard result else {
                                return false
                        }
                }

                state.tHit = FloatX.infinity
                if stopWithRussianRoulette(
                        bounce: state.bounce,
                        pathThroughputWeight: &state.throughput)
                {
                        return false
                }
                return true
        }

        private mutating func bounces(
                state: inout BounceState,
                context: inout IntegratorContext
        ) throws {
                for bounce in state.bounce...maxDepth {
                        state.bounce = bounce
                        let result = try oneBounce(state: &state, context: &context)
                        if !result {
                                break
                        }
                }
        }

        // Deprecated: used by commented out code in surfaceEstimate
        private mutating func bounces(
                state: inout BounceState,
                context: inout IntegratorContext,
                interaction: inout SurfaceInteraction
        ) throws {
                for bounce in state.bounce...maxDepth {
                        state.bounce = bounce
                        state.interaction = interaction
                        let result = try oneBounce(state: &state, context: &context)
                        interaction = state.interaction
                        if !result {
                                break
                        }
                }
        }
}

extension VolumePathIntegrator {
        // HACK: Imagemagick's converts grayscale images to one channel which Intel
        // denoiser can't read. Make white a little colorful
        private func intelHack(_ albedo: inout RgbSpectrum) {
                if albedo.red == albedo.green && albedo.red == albedo.blue {
                        albedo.red += 0.01
                }
        }

}
