import Foundation
import mojoKernel

struct TileRenderer: Renderer {

        init(
                camera: PerspectiveCamera,
                integrator: VolumePathIntegrator,
                sampler: Sampler,
                lightSampler: LightSampler,
                tileSize: (Int, Int),
                immutableState: ImmutableState,
                renderOptions: RenderOptions
        ) {
                self.camera = camera
                self.integrator = integrator
                self.sampler = sampler
                self.lightSampler = lightSampler
                self.tileSize = tileSize
                self.immutableState = immutableState
                self.renderOptions = renderOptions

                let sampleBounds = camera.getSampleBounds()
                if renderOptions.singleRay {
                        let point = sampleBounds.pMin + renderOptions.singleRayCoordinate
                        bounds = Bounds2i(pMin: point, pMax: point + Point2i(x: 1, y: 1))
                } else {
                        bounds = Bounds2i(pMin: sampleBounds.pMin, pMax: sampleBounds.pMax)
                }
        }

        private func generateTiles(from bounds: Bounds2i) -> [Tile] {
                var tiles: [Tile] = []
                var minY = bounds.pMin.y
                while minY < bounds.pMax.y {
                        var minX = bounds.pMin.x
                        while minX < bounds.pMax.x {
                                let pMin = Point2i(x: minX, y: minY)
                                let pMax = Point2i(
                                        x: min(minX + tileSize.0, bounds.pMax.x),
                                        y: min(minY + tileSize.1, bounds.pMax.y))
                                let bounds = Bounds2i(pMin: pMin, pMax: pMax)
                                let tile = Tile(
                                        integrator: integrator,
                                        bounds: bounds)
                                tiles.append(tile)
                                minX += tileSize.0
                        }
                        minY += tileSize.1
                }
                return tiles
        }

        private func renderTile(
                tile: Tile, state: ImmutableState
        ) throws -> (
                samples: [Sample], stats: RenderStats
        ) {
                var tileSampler = self.sampler
                var lightSampler = self.lightSampler
                var tile = tile
                let result = try tile.render(
                        sampler: &tileSampler,
                        camera: self.camera,
                        lightSampler: &lightSampler,
                        state: state
                )
                return result
        }

        // CPU + Sobol + Gaussian path: Mojo renders all tiles in one call (step 15).
        private func renderImageMojo(bounds: Bounds2i) async throws {
                let (rasterToCamera, cameraToWorld) = camera.cameraMatrices()
                let resX = bounds.pMax.x - bounds.pMin.x
                let resY = bounds.pMax.y - bounds.pMin.y
                let pixelCount = resX * resY
                let zero = TileResult_C(
                        estimateR: 0, estimateG: 0, estimateB: 0,
                        albedoR: 0, albedoG: 0, albedoB: 0,
                        filterWeight: 0, pixelX: 0, pixelY: 0)
                var results = [TileResult_C](repeating: zero, count: pixelCount)

                guard case .sobol(let zSobol) = sampler,
                      let gaussianFilter = camera.film.filter as? GaussianFilter else {
                        return
                }
                let fp = gaussianFilter.mojoParams()
                var rng = Xoshiro()
                let rngSeed = UInt64.random(in: 0...UInt64.max, using: &rng)

                let shadeStart = Date()
                integrator.accelerator.renderAllTilesMojo(
                        rasterToCamera: rasterToCamera,
                        cameraToWorld: cameraToWorld,
                        bounds: bounds,
                        tileW: tileSize.0, tileH: tileSize.1,
                        sobolSeed: Int32(zSobol.seed),
                        log2SamplesPerPixel: Int32(zSobol.log2SamplesPerPixel),
                        nBase4Digits: Int32(zSobol.nBase4Digits),
                        samplesPerPixel: Int32(sampler.samplesPerPixel),
                        filterSigma: fp.sigma,
                        filterSupportX: fp.supportX, filterSupportY: fp.supportY,
                        filterNormX: fp.normX, filterNormY: fp.normY,
                        filterWeight: fp.weight,
                        rngSeed: rngSeed,
                        scene: integrator.scene,
                        results: &results,
                        maxDepth: integrator.maxDepth)

                print(String(format: "Telemetry - GPU BVH Time: %.2fs", 0.0))
                print(String(format: "Telemetry - CPU Shade Time: %.2fs",
                             Date().timeIntervalSince(shadeStart)))

                var beautyRGB = [Float](repeating: 0, count: pixelCount * 3)
                var albedoRGB = [Float](repeating: 0, count: pixelCount * 3)
                results.withUnsafeBufferPointer { rPtr in
                        beautyRGB.withUnsafeMutableBufferPointer { bPtr in
                                albedoRGB.withUnsafeMutableBufferPointer { aPtr in
                                        mojo_normalize_film(
                                                rPtr.baseAddress,
                                                Int32(pixelCount),
                                                Float(camera.film.iso),
                                                Float(camera.film.maxComponentValue),
                                                bPtr.baseAddress!,
                                                aPtr.baseAddress!)
                                }
                        }
                }

                var beautyImage = Image(resolution: camera.film.resolution)
                var albedoImage = Image(resolution: camera.film.resolution)
                let normalImage = Image(resolution: camera.film.resolution)
                for i in 0..<pixelCount {
                        let r = results[i]
                        let pixel = Point2i(x: Int(r.pixelX), y: Int(r.pixelY))
                        let bi = i * 3
                        beautyImage.setPixel(
                                color: RgbSpectrum(
                                        red: Real(beautyRGB[bi]),
                                        green: Real(beautyRGB[bi + 1]),
                                        blue: Real(beautyRGB[bi + 2])),
                                atLocation: pixel)
                        albedoImage.setPixel(
                                color: RgbSpectrum(
                                        red: Real(albedoRGB[bi]),
                                        green: Real(albedoRGB[bi + 1]),
                                        blue: Real(albedoRGB[bi + 2])),
                                atLocation: pixel)
                }
                try await camera.film.writeNormalizedImages(
                        beauty: &beautyImage, albedo: albedoImage, normal: normalImage,
                        tileSize: tileSize)
        }

        private func renderImage(bounds: Bounds2i, immutableState: ImmutableState) async throws {
                // Fast path: Sobol sampler + Gaussian filter + no GPU → single Mojo call.
                if integrator.accelerator.gpuSceneHandle == nil,
                   case .sobol = sampler,
                   camera.film.filter is GaussianFilter {
                        try await renderImageMojo(bounds: bounds)
                        return
                }

                // Fallback path (GPU or non-Sobol sampler): tile dispatch via Swift task group.
                let tiles = generateTiles(from: bounds)
                let reporter = ProgressReporter(title: "Rendering", total: tiles.count)
                async let progressTask: Void = runProgressReporter(reporter: reporter)
                let maxConcurrent = max(1, ProcessInfo.processInfo.activeProcessorCount - 1)

                try await withThrowingTaskGroup(of: (samples: [Sample], stats: RenderStats).self) { group in
                        var tileIterator = tiles.makeIterator()
                        for _ in 0..<maxConcurrent {
                                guard let tile = tileIterator.next() else { break }
                                group.addTask {
                                        let result = try self.renderTile(tile: tile, state: immutableState)
                                        await reporter.tileFinished()
                                        return result
                                }
                        }
                        var allSamples: [Sample] = []
                        var totalStats = RenderStats()
                        for try await result in group {
                                allSamples.append(contentsOf: result.samples)
                                totalStats.bvhTime += result.stats.bvhTime
                                totalStats.shadeTime += result.stats.shadeTime
                                if let tile = tileIterator.next() {
                                        group.addTask {
                                                let result = try self.renderTile(tile: tile, state: immutableState)
                                                await reporter.tileFinished()
                                                return result
                                        }
                                }
                        }
                        print(String(format: "Telemetry - GPU BVH Time: %.2fs", totalStats.bvhTime))
                        print(String(format: "Telemetry - CPU Shade Time: %.2fs", totalStats.shadeTime))
                        try await self.camera.film.writeImages(samples: allSamples, tileSize: tileSize)
                }
                await progressTask
        }

        func render() async throws {
                let timer = Timer("Rendering...")
                try await renderImage(bounds: bounds, immutableState: immutableState)
                let output = "\n\n" + timer.elapsed + "\n"
                FileHandle.standardOutput.write(Data(output.utf8))
        }

        let camera: PerspectiveCamera
        let integrator: VolumePathIntegrator
        let immutableState: ImmutableState
        let lightSampler: LightSampler
        let renderOptions: RenderOptions
        let sampler: Sampler
        let bounds: Bounds2i
        let tileSize: (Int, Int)
}
