import Foundation

final class Renderer {

        init(
                hierarchy: BoundingHierarchy,
                camera: Camera,
                integrator: PathIntegrator,
                sampler: Sampler,
                scene: Scene
        ) {
                self.camera = camera
                self.integrator = integrator
                self.sampler = sampler
                self.scene = scene
                self.hierarchy = hierarchy
        }

        func generateTiles(from bounds: Bounds2i) -> [Tile] {
                var tiles: [Tile] = []
                var y = bounds.pMin.y
                while y < bounds.pMax.y {
                        var x = bounds.pMin.x
                        while x < bounds.pMax.x {
                                let pMin = Point2I(x: x, y: y)
                                let pMax = Point2I(
                                        x: min(x + Tile.size, bounds.pMax.x),
                                        y: min(y + Tile.size, bounds.pMax.y))
                                let bounds = Bounds2i(pMin: pMin, pMax: pMax)
                                let tile = Tile(
                                        integrator: integrator,
                                        bounds: bounds)
                                tiles.append(tile)
                                x += Tile.size
                        }
                        y += Tile.size
                }
                return tiles
        }

        func renderTile(tile: Tile) throws -> [Sample] {
                let tileSampler = self.sampler.clone()
                return try tile.render(
                        reporter: reporter,
                        sampler: tileSampler,
                        camera: self.camera,
                        hierarchy: hierarchy
                )
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

        func doRenderTile(tile: Tile) throws {
                if renderSynchronously {
                        try renderSync(tile: tile)
                } else {
                        renderAsync(tile: tile)
                }
        }

        func generateBounds() -> Bounds2i {
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

        func renderTiles(tiles: [Tile]) throws {
                for tile in tiles {
                        try doRenderTile(tile: tile)
                }
        }

        func renderImage(bounds: Bounds2i) throws {
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
        let hierarchy: BoundingHierarchy
        let integrator: PathIntegrator
        let queue = DispatchQueue.global()
        var reporter = ProgressReporter()
        let sampler: Sampler
        let scene: Scene
}
