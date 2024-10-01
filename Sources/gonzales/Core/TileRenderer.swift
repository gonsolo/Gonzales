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
                        bounds = Bounds2i(pMin: point, pMax: point + Point2I(x: 1, y: 1))
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
                                let pMin = Point2I(x: x, y: y)
                                let pMax = Point2I(
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

        private func renderTile(tile: Tile) async throws -> [Sample] {
                let tileSampler = await self.sampler.clone()
                let samples = try await tile.render(
                        reporter: reporter,
                        sampler: tileSampler,
                        camera: self.camera,
                        scene: scene,
                        lightSampler: lightSampler
                )
                return samples
        }

        private func renderImage(bounds: Bounds2i) async throws {
                let tiles = generateTiles(from: bounds)
                await withThrowingTaskGroup(of: Void.self) { group in
                        for tile in tiles {
                                group.addTask {
                                        let samples = try await self.renderTile(tile: tile)
                                        await self.camera.film.add(samples: samples)
                                }
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
