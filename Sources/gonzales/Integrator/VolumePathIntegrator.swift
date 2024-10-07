// Path tracing
// "James Kajiya: The Rendering Equation"
// DOI: 10.1145/15922.15902

final class VolumePathIntegrator: Sendable {

        init(scene: Scene, maxDepth: Int) {
                self.scene = scene
                self.maxDepth = maxDepth
        }

        private func brdfDensity(
                light: Light,
                interaction: Interaction,
                sample: Vector
        ) -> FloatX {
                return interaction.evaluateProbabilityDensity(wi: sample)
        }

        @MainActor
        private func chooseLight(
                sampler: Sampler,
                lightSampler: LightSampler
        ) async throws
                -> (Light, FloatX)
        {
                return await lightSampler.chooseLight()
        }

        private func intersectOrInfiniteLights(
                rays: [Ray],
                tHits: inout [FloatX],
                bounce: Int,
                estimates: inout [RgbSpectrum],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                for i in 0..<rays.count {
                        if !skips[i] {
                                interactions[i].valid = false
                        }
                }
                try scene.intersect(
                        rays: rays,
                        tHits: &tHits,
                        interactions: &interactions,
                        skips: skips)
                for i in 0..<rays.count {
                        if interactions[i].valid {
                                continue
                        }
                        let radiance = scene.infiniteLights.reduce(
                                black,
                                { accumulated, light in accumulated + light.radianceFromInfinity(for: rays[i])
                                }
                        )
                        if bounce == 0 { estimates[i] += radiance }
                }
        }

        @MainActor
        private func sampleLightSource(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> BsdfSample {
                let (radiance, wi, lightDensity, visibility) = await light.sample(
                        for: interaction, u: sampler.get2D())
                guard !radiance.isBlack && !lightDensity.isInfinite else {
                        return invalidBsdfSample
                }
                guard try await visibility.unoccluded(scene: scene) else {
                        return invalidBsdfSample
                }
                let scatter = await interaction.evaluateDistributionFunction(wi: wi)
                let estimate = scatter * radiance
                return BsdfSample(estimate, wi, lightDensity)
        }

        @MainActor
        private func sampleDistributionFunction(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> BsdfSample {

                let zero = BsdfSample()
                var bsdfSample = await interaction.sampleDistributionFunction(sampler: sampler)
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

        @MainActor
        private func sampleLight(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> RgbSpectrum {
                let lightSample = try await sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                if lightSample.probabilityDensity == 0 {
                        return black
                } else {
                        return lightSample.estimate / lightSample.probabilityDensity
                }
        }

        @MainActor
        private func sampleGlobalBsdf(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> RgbSpectrum {
                let bsdfSample = try await sampleDistributionFunction(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                if bsdfSample.probabilityDensity == 0 {
                        return black
                } else {
                        return bsdfSample.estimate / bsdfSample.probabilityDensity
                }
        }

        @MainActor
        private func sampleMultipleImportance(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> RgbSpectrum {

                let lightSample = try await sampleLightSource(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                let brdfDensity = brdfDensity(
                        light: light,
                        interaction: interaction,
                        sample: lightSample.incoming)
                let lightWeight = powerHeuristic(f: lightSample.probabilityDensity, g: brdfDensity)
                let a =
                        lightSample.probabilityDensity == 0
                        ? black : lightSample.estimate * lightWeight / lightSample.probabilityDensity

                let brdfSample = try await sampleDistributionFunction(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                let lightDensity = try await light.probabilityDensityFor(
                        samplingDirection: brdfSample.incoming,
                        from: interaction)
                let brdfWeight = powerHeuristic(f: brdfSample.probabilityDensity, g: lightDensity)
                let b =
                        brdfSample.probabilityDensity == 0
                        ? black : brdfSample.estimate * brdfWeight / brdfSample.probabilityDensity

                return a + b
        }

        @MainActor
        private func estimateDirect(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) async throws -> RgbSpectrum {
                if light.isDelta {
                        return try await sampleLight(
                                light: light,
                                interaction: interaction,
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

                return try await sampleMultipleImportance(
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
        }

        @MainActor
        private func sampleOneLight(
                at interaction: Interaction,
                with sampler: Sampler,
                lightSampler: LightSampler
        ) async throws -> RgbSpectrum {
                let (light, lightPdf) = try await chooseLight(
                        sampler: sampler,
                        lightSampler: lightSampler)
                let estimate = try await estimateDirect(
                        light: light,
                        interaction: interaction,
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

        @MainActor
        private func sampleMedium(
                pathThroughputWeight: RgbSpectrum,
                mediumInteraction: MediumInteraction,
                sampler: Sampler,
                lightSampler: LightSampler,
                ray: Ray
        ) async throws -> (RgbSpectrum, Ray) {
                let estimate =
                        try await pathThroughputWeight
                        * sampleOneLight(
                                at: mediumInteraction,
                                with: sampler,
                                lightSampler: lightSampler)
                let (_, wi) = await mediumInteraction.phase.samplePhase(
                        wo: -ray.direction,
                        sampler: sampler)
                let spawnedRay = mediumInteraction.spawnRay(inDirection: wi)
                return (estimate, spawnedRay)
        }

        @MainActor
        func oneBounce(
                interactions: inout [SurfaceInteraction],
                tHits: inout [Float],
                rays: inout [Ray],
                bounce: Int,
                estimates: inout [RgbSpectrum],
                sampler: Sampler,
                pathThroughputWeights: inout [RgbSpectrum],
                lightSampler: LightSampler,
                albedos: inout [RgbSpectrum],
                firstNormals: inout [Normal],
                skips: [Bool]
        ) async throws -> [Bool] {
                var results = Array(repeating: false, count: rays.count)
                try intersectOrInfiniteLights(
                        rays: rays,
                        tHits: &tHits,
                        bounce: bounce,
                        estimates: &estimates,
                        interactions: &interactions,
                        skips: skips)
                for i in 0..<rays.count {
                        if skips[i] {
                                continue
                        }
                        if !interactions[i].valid {
                                results[i] = false
                        } else {
                                results[i] = try await oneBounce(
                                        interaction: &interactions[i],
                                        tHit: &tHits[i],
                                        ray: &rays[i],
                                        bounce: bounce,
                                        estimate: &estimates[i],
                                        sampler: sampler,
                                        pathThroughputWeight: &pathThroughputWeights[i],
                                        lightSampler: lightSampler,
                                        albedo: &albedos[i],
                                        firstNormal: &firstNormals[i],
                                        skip: skips[i]
                                )
                        }
                }
                return results
        }

        func mediumEstimate(
                ray: inout Ray,
                pathThroughputWeight: RgbSpectrum,
                mediumInteraction: MediumInteraction,
                sampler: Sampler,
                lightSampler: LightSampler
        ) async throws -> RgbSpectrum {
                var mediumRadiance = black
                (mediumRadiance, ray) = try await sampleMedium(
                        pathThroughputWeight: pathThroughputWeight,
                        mediumInteraction: mediumInteraction,
                        sampler: sampler,
                        lightSampler: lightSampler,
                        ray: ray)
                return mediumRadiance
        }

        @MainActor
        func surfaceEstimate(
                interaction: inout SurfaceInteraction,
                ray: inout Ray,
                bounce: Int,
                pathThroughputWeight: inout RgbSpectrum,
                estimate: inout RgbSpectrum,
                tHit: inout Float,
                sampler: Sampler,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal
        ) async throws -> Bool {
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
                        return false
                }
                if materials[surfaceInteraction.material].isInterface {
                        var spawnedRay = surfaceInteraction.spawnRay(
                                inDirection: ray.direction)
                        if let interface = surfaceInteraction.mediumInterface {
                                spawnedRay.medium = state.namedMedia[interface.interior]
                        }
                        ray = spawnedRay
                        try await bounces(
                                ray: &ray,
                                interaction: &interaction,
                                tHit: &tHit,
                                bounce: bounce + 1,
                                estimate: &estimate,
                                sampler: sampler,
                                pathThroughputWeight: &pathThroughputWeight,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal)
                }
                surfaceInteraction.setBsdf()
                if bounce == 0 {
                        albedo = surfaceInteraction.bsdf.albedo()
                        firstNormal = surfaceInteraction.normal
                }
                let lightEstimate =
                        try await pathThroughputWeight
                        * sampleOneLight(
                                at: surfaceInteraction,
                                with: sampler,
                                lightSampler: lightSampler)
                estimate += lightEstimate
                let (bsdfSample, _) = await surfaceInteraction.bsdf.sampleWorld(
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

        @MainActor
        func oneBounce(
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                ray: inout Ray,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: Sampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal,
                skip: Bool
        ) async throws -> Bool {
                let (transmittance, mediumInteraction) =
                        await ray.medium?.sample(ray: ray, tHit: tHit, sampler: sampler) ?? (white, nil)
                pathThroughputWeight *= transmittance
                if pathThroughputWeight.isBlack {
                        return false
                }
                guard bounce < maxDepth else {
                        return false
                }
                if let mediumInteraction {
                        estimate += try await mediumEstimate(
                                ray: &ray,
                                pathThroughputWeight: pathThroughputWeight,
                                mediumInteraction: mediumInteraction,
                                sampler: sampler,
                                lightSampler: lightSampler)
                } else {
                        let result = try await surfaceEstimate(
                                interaction: &interaction,
                                ray: &ray,
                                bounce: bounce,
                                pathThroughputWeight: &pathThroughputWeight,
                                estimate: &estimate,
                                tHit: &tHit,
                                sampler: sampler,
                                lightSampler: lightSampler,
                                albedo: &albedo,
                                firstNormal: &firstNormal)
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
                rays: inout [Ray],
                interactions: inout [SurfaceInteraction],
                tHits: inout [Float],
                bounce: Int,
                estimates: inout [RgbSpectrum],
                sampler: Sampler,
                pathThroughputWeights: inout [RgbSpectrum],
                lightSampler: LightSampler,
                albedos: inout [RgbSpectrum],
                firstNormals: inout [Normal]
        ) async throws {
                var skip = Array(repeating: false, count: rays.count)
                for bounce in bounce...maxDepth {
                        let results = try await oneBounce(
                                interactions: &interactions,
                                tHits: &tHits,
                                rays: &rays,
                                bounce: bounce,
                                estimates: &estimates,
                                sampler: sampler,
                                pathThroughputWeights: &pathThroughputWeights,
                                lightSampler: lightSampler,
                                albedos: &albedos,
                                firstNormals: &firstNormals,
                                skips: skip
                        )
                        for i in 0..<rays.count {
                                if !results[i] {
                                        skip[i] = true
                                }
                        }
                }
        }

        func bounces(
                ray: inout Ray,
                interaction: inout SurfaceInteraction,
                tHit: inout Float,
                bounce: Int,
                estimate: inout RgbSpectrum,
                sampler: Sampler,
                pathThroughputWeight: inout RgbSpectrum,
                lightSampler: LightSampler,
                albedo: inout RgbSpectrum,
                firstNormal: inout Normal
        ) async throws {
                var skip = false
                for bounce in bounce...maxDepth {
                        let result = try await oneBounce(
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
                                skip: skip
                        )
                        if !result {
                                skip = true
                        }
                }
        }

        func getRadiancesAndAlbedos(
                from rays: [Ray],
                tHits: inout [FloatX],
                with sampler: Sampler,
                lightSampler: LightSampler
        ) async throws
                -> [(estimate: RgbSpectrum, albedo: RgbSpectrum, normal: Normal)]
        {

                // Path throughput weight
                // The product of all GlobalBsdfs and cosines divided by the pdf
                // Π f |cosθ| / pdf
                //var pathThroughputWeight = white
                var pathThroughputWeights = Array(repeating: white, count: rays.count)

                var estimates = Array(repeating: black, count: rays.count)
                var varRays = rays
                var albedos = Array(repeating: black, count: rays.count)
                var firstNormals = Array(repeating: zeroNormal, count: rays.count)
                var interactions = Array(repeating: SurfaceInteraction(), count: rays.count)
                try await bounces(
                        rays: &varRays,
                        interactions: &interactions,
                        tHits: &tHits,
                        bounce: 0,
                        estimates: &estimates,
                        sampler: sampler,
                        pathThroughputWeights: &pathThroughputWeights,
                        lightSampler: lightSampler,
                        albedos: &albedos,
                        firstNormals: &firstNormals)

                //intelHack(&albedo)
                var estimatesAlbedosNormals = [(RgbSpectrum, RgbSpectrum, Normal)]()
                for i in 0..<rays.count {
                        let estimateAlbedoNormal = (
                                estimate: estimates[i],
                                albedo: albedos[i],
                                normal: firstNormals[i]
                        )
                        estimatesAlbedosNormals.append(estimateAlbedoNormal)
                }
                //}
                return estimatesAlbedosNormals
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
