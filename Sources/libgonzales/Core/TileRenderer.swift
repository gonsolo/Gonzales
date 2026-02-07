@preconcurrency import Foundation

struct TileRenderer: Renderer {

        @MainActor
        init(
                camera: PerspectiveCamera,
                integrator: VolumePathIntegrator,
                sampler: Sampler,
                lightSampler: LightSampler,
                tileSize: (Int, Int)
        ) async {
                self.camera = camera
                self.integrator = integrator
                self.sampler = sampler
                self.lightSampler = lightSampler
                self.tileSize = tileSize

                let sampleBounds = await camera.getSampleBounds()
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

        private func renderTile(tile: Tile, state: ImmutableState) throws -> [Sample] {
                var tileSampler = self.sampler
                var lightSampler = self.lightSampler
                var tile = tile
                let samples = try tile.render(
                        sampler: &tileSampler,
                        camera: self.camera,
                        lightSampler: &lightSampler,
                        state: state
                )
                return samples
        }

        @MainActor
        func runProgressReporter(reporter: ProgressReporter) async {
                let reportInterval: Duration = .milliseconds(500)

                // Helper function using only basic types and math
                func formatTime(_ interval: TimeInterval) -> String {
                        let interval = Int(interval.rounded())
                        let seconds = interval % 60
                        let minutes = (interval / 60) % 60
                        let hours = interval / 3600

                        if hours > 0 {
                                return String(format: "%dh %dm %ds", hours, minutes, seconds)
                        } else if minutes > 0 {
                                return String(format: "%dm %ds", minutes, seconds)
                        } else {
                                return String(format: "%.1fs", Double(seconds))
                        }
                }

                while !Task.isCancelled {
                        do {
                                try await Task.sleep(for: reportInterval)

                                let metrics = await reporter.getProgressMetrics()

                                // Calculate Estimated Total Time (ETT)
                                let predictedTotalTime = metrics.averageTimePerTile * Double(metrics.total)

                                let percentage = (Double(metrics.completed) / Double(metrics.total)) * 100.0

                                let progressString = String(
                                        format:
                                                "Progress: %d / %d (%.1f%%) | Elapsed: %@ | Total Est.: %@",
                                        metrics.completed,
                                        metrics.total,
                                        percentage,
                                        formatTime(metrics.timeElapsed),
                                        formatTime(max(0, predictedTotalTime))  // Display ETT
                                )

                                // Periodic Update
                                print(progressString, terminator: "\r")
                                fflush(stdout)

                                if metrics.completed >= metrics.total {
                                        break
                                }
                        } catch {
                                break
                        }
                }

                // Final output cleanup
                let finalMetrics = await reporter.getProgressMetrics()
                if finalMetrics.completed == finalMetrics.total {
                        let finalString =
                                "Progress: \(finalMetrics.total) / \(finalMetrics.total) (100.0%) - "
                                + "Rendering Complete. Total Time: \(formatTime(finalMetrics.timeElapsed))\n"

                        print(finalString, terminator: "")
                        fflush(stdout)
                }
        }

        @MainActor
        private func renderImage(bounds: Bounds2i) async throws {
                let immutableState = state.getImmutable()
                let tiles = generateTiles(from: bounds)
                let reporter = ProgressReporter(total: tiles.count)

                async let _: Void = runProgressReporter(reporter: reporter)

                let renderingQueue = DispatchQueue(
                        label: "com.renderer.heavy-work", qos: .userInitiated, attributes: .concurrent)

                try await withThrowingTaskGroup(of: [Sample].self) { group in
                        for tile in tiles {

                                group.addTask {

                                        return try await withCheckedThrowingContinuation { continuation in

                                                renderingQueue.async {
                                                        do {
                                                                let samples = try self.renderTile(
                                                                        tile: tile, state: immutableState)

                                                                Task {
                                                                        await reporter.tileFinished()
                                                                }

                                                                continuation.resume(returning: samples)

                                                        } catch {
                                                                continuation.resume(throwing: error)
                                                        }
                                                }
                                        }
                                }
                        }

                        var allSamples: [Sample] = []
                        for try await samples in group {
                                allSamples.append(contentsOf: samples)
                        }

                        try await self.camera.film.writeImages(samples: allSamples, tileSize: tileSize)
                }
        }
        @MainActor
        func render() async throws {
                let timer = Timer("Rendering...")
                try await renderImage(bounds: bounds)
                print("\n")
                print(timer.elapsed)
                fflush(stdout)
        }

        let camera: PerspectiveCamera
        let integrator: VolumePathIntegrator
        let lightSampler: LightSampler
        let sampler: Sampler
        let bounds: Bounds2i
        let tileSize: (Int, Int)
}
