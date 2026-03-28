import Foundation

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

        private func renderTile(tile: Tile, state: ImmutableState) throws -> (samples: [Sample], stats: RenderStats) {
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

        private func renderImage(bounds: Bounds2i, immutableState: ImmutableState) async throws {
                let tiles = generateTiles(from: bounds)
                let reporter = ProgressReporter(title: "Rendering", total: tiles.count)

                async let progressTask: Void = runProgressReporter(reporter: reporter)

                // Limit concurrency to keep one cooperative thread free for the progress reporter.
                let maxConcurrent = max(1, ProcessInfo.processInfo.activeProcessorCount - 1)

                try await withThrowingTaskGroup(of: (samples: [Sample], stats: RenderStats).self) { group in
                        var tileIterator = tiles.makeIterator()

                        // Seed the group with maxConcurrent tasks
                        for _ in 0..<maxConcurrent {
                                guard let tile = tileIterator.next() else { break }
                                group.addTask {
                                        let result = try self.renderTile(tile: tile, state: immutableState)
                                        await reporter.tileFinished()
                                        return result
                                }
                        }

                        // As each task completes, add the next tile
                        var allSamples: [Sample] = []
                        var totalStats = RenderStats()

                        for try await result in group {
                                allSamples.append(contentsOf: result.samples)
                                totalStats.bvhTime += result.stats.bvhTime
                                totalStats.shadeTime += result.stats.shadeTime

                                if let tile = tileIterator.next() {
                                        group.addTask {
                                                let result = try self.renderTile(
                                                        tile: tile, state: immutableState)
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
