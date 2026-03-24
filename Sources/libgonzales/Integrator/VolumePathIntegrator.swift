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
                tHit: inout Real,
                with sampler: inout Sampler,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws -> RayTraceSample {

                // Path throughput weight
                // The product of all FramedBsdfs and cosines divided by the pdf
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
                var tHit: Real
                var bounce: Int
                var estimate: RgbSpectrum
                var throughput: RgbSpectrum
                var albedo: RgbSpectrum
                var firstNormal: Normal
                var interaction: SurfaceInteraction?
                var specularBounce: Bool = false
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
        ) -> Real {
                return distributionModel.evaluateProbabilityDensity(outgoing: outgoing, incident: sample)
        }

        private func chooseLight(
                sampler _: Sampler,
                lightSampler: inout LightSampler,
                scene: Scene
        ) throws
                -> (Light, Real) {
                return try lightSampler.chooseLight(scene: scene)
        }

        private func sampleLightSource<I: Interaction, D: DistributionModel>(
                light: Light,
                interaction: I,
                distributionModel: D,
                sampler: inout Sampler,
                scene: Scene
        ) throws -> BsdfSample {
                let lightSample = try light.sample(
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
                var tHit = Real.infinity
                let brdfInteraction = try accelerator.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
                if let brdfInteraction {
                        if let areaLight = brdfInteraction.areaLight {
                                let radiance = areaLight.emittedRadiance(
                                        from: brdfInteraction,
                                        inDirection: bsdfSample.incoming)
                                bsdfSample.estimate *= radiance
                                return bsdfSample
                        }
                        return zero
                }
                for light in scene.lights {
                        switch light {
                        case .infinite(let infiniteLight):
                                let radiance = infiniteLight.radianceFromInfinity(
                                        for: ray, arena: scene.arena)
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

        private func sampleFramedBsdf<I: Interaction, D: DistributionModel>(
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

        // @doc:mis-sampling
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
        // @doc:end

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
                // return try sampleFramedBsdf(
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

        // @doc:russian-roulette
        private mutating func stopWithRussianRoulette(bounce: Int, pathThroughputWeight: inout RgbSpectrum)
                -> Bool {
                if pathThroughputWeight.maxValue < 1 && bounce > 1 {
                        let probability: Real = max(0, 1 - pathThroughputWeight.maxValue)
                        let roulette = Real.random(in: 0..<1, using: &xoshiro)
                        if roulette < probability {
                                return true
                        }
                        pathThroughputWeight /= 1 - probability
                }
                return false
        }
        // @doc:end

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
                guard let surfaceInteraction = state.interaction else { return false }
                if state.bounce == 0 || state.specularBounce {
                        if let areaLight = surfaceInteraction.areaLight {
                                state.estimate +=
                                        state.throughput
                                        * areaLight.emittedRadiance(
                                                from: surfaceInteraction,
                                                inDirection: surfaceInteraction.outgoing)
                        }
                }
                assert(surfaceInteraction.materialIndex >= 0)
                let bsdf = try scene.materials[surfaceInteraction.materialIndex].getBsdf(
                        interaction: surfaceInteraction, arena: scene.arena)

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
                let (bsdfSample, _) = bsdf.sampleWorldSpace(
                        outgoing: surfaceInteraction.outgoing, uSample: context.sampler.get3D())
                guard
                        bsdfSample.probabilityDensity != 0
                                && !bsdfSample.probabilityDensity.isNaN
                else {
                        return false
                }
                state.specularBounce = bsdf.isSpecular
                state.throughput *= bsdfSample.throughputWeight(normal: surfaceInteraction.normal)
                let spawnedRay = surfaceInteraction.spawnRay(inDirection: bsdfSample.incoming)
                state.ray = spawnedRay
                return true
        }

        private mutating func oneBounce(
                state: inout BounceState,
                context: inout IntegratorContext
        ) throws -> Bool {

                state.interaction = try accelerator.intersect(
                        scene: scene,
                        ray: state.ray,
                        tHit: &state.tHit)

                if state.interaction == nil {
                        if state.bounce == 0 || state.specularBounce {
                                let radiance = scene.infiniteLights.reduce(
                                        black,
                                        { accumulated, light in
                                                accumulated
                                                        + light.radianceFromInfinity(
                                                                for: state.ray, arena: scene.arena)
                                        }
                                )
                                state.estimate += state.throughput * radiance
                        }
                        return false  // No surface hit, so stop this bounce
                }
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

                state.tHit = Real.infinity
                if stopWithRussianRoulette(
                        bounce: state.bounce,
                        pathThroughputWeight: &state.throughput) {
                        return false
                }
                return true
        }

        // @doc:bounce-loop
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
        // @doc:end
}
