@preconcurrency import Foundation

final class TileRenderer: Renderer {

        init(
                accelerator: Accelerator,
                camera: Camera,
                integrator: VolumePathIntegrator,
                sampler: Sampler,
                scene: Scene,
                lightSampler: LightSampler,
                tileSize: (Int, Int)
        ) async {
                self.camera = camera
                self.integrator = integrator
                self.sampler = sampler
                self.scene = scene
                self.accelerator = accelerator
                self.lightSampler = lightSampler
                self.tileSize = tileSize

                let sampleBounds = await camera.getSampleBounds()
                if await singleRay {
                        let point = await sampleBounds.pMin + singleRayCoordinate
                        bounds = Bounds2i(pMin: point, pMax: point + Point2i(x: 1, y: 1))
                } else {
                        bounds = Bounds2i(pMin: sampleBounds.pMin, pMax: sampleBounds.pMax)
                }
                reporter = ProgressReporter(total: bounds.area() * sampler.samplesPerPixel)
        }

        private func generateTiles(from bounds: Bounds2i) -> [Tile] {
                var tiles: [Tile] = []
                var y = bounds.pMin.y
                while y < bounds.pMax.y {
                        var x = bounds.pMin.x
                        while x < bounds.pMax.x {
                                let pMin = Point2i(x: x, y: y)
                                let pMax = Point2i(
                                        x: min(x + tileSize.0, bounds.pMax.x),
                                        y: min(y + tileSize.1, bounds.pMax.y))
                                let bounds = Bounds2i(pMin: pMin, pMax: pMax)
                                let tile = Tile(
                                        integrator: integrator,
                                        bounds: bounds)
                                tiles.append(tile)
                                x += tileSize.0
                        }
                        y += tileSize.1
                }
                return tiles
        }

        private func renderTile(tile: Tile, state: ImmutableState) async throws -> [Sample] {
                let tileSampler = self.sampler.clone()
                let samples = try tile.render(
                        reporter: reporter,
                        sampler: tileSampler,
                        camera: self.camera,
                        scene: scene,
                        lightSampler: lightSampler,
                        state: state
                )
                return samples
        }

        @MainActor
        private func renderImage(bounds: Bounds2i) async throws {
                let immutableState = state.getImmutable()
                let tiles = generateTiles(from: bounds)
                try await withThrowingTaskGroup(of: [Sample].self) { group in
                        for tile in tiles {
                                group.addTask {
                                        return try await self.renderTile(tile: tile, state: immutableState)
                                }
                        }
                        for try await samples in group {
                                await self.camera.film.add(samples: samples)
                        }
                }
        }

        @MainActor
        func render() async throws {
                let timer = Timer("Rendering...")
                await reporter.reset()
                try await renderImage(bounds: bounds)
                try await camera.film.writeImages()
                print("\n")
                print(timer.elapsed)
                fflush(stdout)
        }

        let camera: Camera
        let accelerator: Accelerator
        let integrator: VolumePathIntegrator
        let lightSampler: LightSampler
        let reporter: ProgressReporter
        let sampler: Sampler
        let scene: Scene
        let bounds: Bounds2i
        let tileSize: (Int, Int)
}
