import Foundation

final class TileRenderer: Renderer {

        init(
                accelerator: Accelerator,
                camera: Camera,
                integrator: VolumePathIntegrator,
                sampler: Sampler,
                scene: Scene,
                lightSampler: LightSampler,
                tileSize: (Int, Int)
        ) {
                self.camera = camera
                self.integrator = integrator
                self.sampler = sampler
                self.scene = scene
                self.accelerator = accelerator
                self.lightSampler = lightSampler
                self.tileSize = tileSize
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

        private func renderTile(tile: Tile) throws -> [Sample] {
                let tileSampler = self.sampler.clone()
                let samples = try tile.render(
                        reporter: reporter,
                        sampler: tileSampler,
                        camera: self.camera,
                        scene: scene,
                        lightSampler: lightSampler
                )
                return samples
        }

        private func renderAndMergeTile(tile: Tile) throws {
                let samples = try renderTile(tile: tile)
                camera.film.add(samples: samples)
        }

        private func renderSync(tile: Tile) throws {
                try queue.sync {
                        try renderAndMergeTile(tile: tile)
                }
        }

        private func renderAsync(tile: Tile) {
                queue.async(group: group) {
                        do {
                                try self.renderAndMergeTile(tile: tile)
                        } catch let error {
                                handle(error)
                                fatalError("in async")
                        }
                }
        }

        private func doRenderTile(tile: Tile) throws {
                if renderSynchronously {
                        try renderSync(tile: tile)
                } else {
                        renderAsync(tile: tile)
                }
        }

        private func generateBounds() -> Bounds2i {
                let sampleBounds = camera.film.getSampleBounds()
                var bounds: Bounds2i
                if singleRay {
                        let point = sampleBounds.pMin + singleRayCoordinate
                        bounds = Bounds2i(pMin: point, pMax: point + Point2I(x: 1, y: 1))
                } else {
                        bounds = Bounds2i(pMin: sampleBounds.pMin, pMax: sampleBounds.pMax)
                }
                return bounds
        }

        private func renderTiles(tiles: [Tile]) throws {
                for tile in tiles {
                        try doRenderTile(tile: tile)
                }
        }

        private func renderImage(bounds: Bounds2i) throws {
                let tiles = generateTiles(from: bounds)
                try renderTiles(tiles: tiles)
        }

        func render() throws {
                let timer = Timer("Rendering...")
                let bounds = generateBounds()
                reporter = ProgressReporter(total: bounds.area() * sampler.samplesPerPixel)
                reporter.reset()
                try renderImage(bounds: bounds)
                group.wait()
                try camera.film.writeImages()
                print("\n")
                print(timer.elapsed)
                fflush(stdout)
        }

        let camera: Camera
        let group = DispatchGroup()
        let accelerator: Accelerator
        let integrator: VolumePathIntegrator
        let lightSampler: LightSampler
        let queue = DispatchQueue.global()
        var reporter = ProgressReporter()
        let sampler: Sampler
        let scene: Scene

        let tileSize: (Int, Int)
}
