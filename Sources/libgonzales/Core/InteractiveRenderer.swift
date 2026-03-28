import Foundation
import vulkanViewer

/// Renders one sample per pixel and displays the result in a Vulkan viewer window.
/// The window stays open until the user closes it. Camera controls (WASD + mouse)
/// allow navigation; on camera change the image is re-rendered at 1 SPP.
struct InteractiveRenderer: Renderer {

        let camera: PerspectiveCamera
        let integrator: VolumePathIntegrator
        let sampler: Sampler
        let lightSampler: LightSampler
        let tileSize: (Int, Int)
        let immutableState: ImmutableState

        func render() async throws {
                let resolution = camera.film.getResolution()
                let width = resolution.x
                let height = resolution.y

                guard let viewer = viewer_create(
                        Int32(width), Int32(height), "Gonzales"
                ) else {
                        print("Error: Failed to create viewer window")
                        return
                }

                // Initialize viewer camera from Gonzales scene camera
                let origin = camera.objectToWorld * Point(x: 0, y: 0, z: 0)
                let dir = camera.objectToWorld * Vector(x: 0, y: 0, z: 1)
                let up = camera.objectToWorld * Vector(x: 0, y: 1, z: 0)

                let initialState = CameraState(
                        posX: Float(origin.x), posY: Float(origin.y), posZ: Float(origin.z),
                        dirX: Float(dir.x), dirY: Float(dir.y), dirZ: Float(dir.z),
                        upX: Float(up.x), upY: Float(up.y), upZ: Float(up.z),
                        cameraChanged: 0
                )
                viewer_set_camera_state(viewer, initialState)

                print("Interactive mode: \(width)×\(height), 1 spp")
                print("Controls: WASD=move, Q/E=up/down, RMB+drag=look, Scroll=speed, Esc=quit")

                // Render 1 SPP and display
                let samples = try renderOneSpp()
                var image = buildImage(from: samples, resolution: resolution)
                pushToViewer(viewer: viewer, image: image, width: width, height: height)

                // Main loop: poll events, re-render on camera change
                while viewer_should_close(viewer) == 0 {
                        viewer_poll_events(viewer)

                        let cameraState = viewer_get_camera_state(viewer)
                        if cameraState.cameraChanged != 0 {
                                let eye = Point(
                                        x: Real(cameraState.posX),
                                        y: Real(cameraState.posY),
                                        z: Real(cameraState.posZ))
                                let target = Point(
                                        x: Real(cameraState.posX + cameraState.dirX),
                                        y: Real(cameraState.posY + cameraState.dirY),
                                        z: Real(cameraState.posZ + cameraState.dirZ))
                                let upVec = Vector(
                                        x: Real(cameraState.upX),
                                        y: Real(cameraState.upY),
                                        z: Real(cameraState.upZ))

                                if let viewTransform = try? lookAtTransform(eye: eye, target: target, upVector: upVec) {
                                        camera.objectToWorld = viewTransform.inverse
                                }

                                let newSamples = try renderOneSpp()
                                image = buildImage(from: newSamples, resolution: resolution)
                                pushToViewer(
                                        viewer: viewer, image: image,
                                        width: width, height: height)
                        }

                        // Sleep briefly to avoid busy-waiting
                        try await Task.sleep(nanoseconds: 16_000_000)  // ~60 fps poll
                }

                viewer_destroy(viewer)
                print("Viewer closed")
        }

        private func renderOneSpp() throws -> [Sample] {
                let bounds = camera.getSampleBounds()
                var tiles = generateTiles(from: bounds)
                var allSamples = [Sample]()

                for i in 0..<tiles.count {
                        var tileSampler = makeSingleSppSampler()
                        var tileLightSampler = self.lightSampler
                        let samples = try tiles[i].render(
                                sampler: &tileSampler,
                                camera: camera,
                                lightSampler: &tileLightSampler,
                                state: immutableState)
                        allSamples.append(contentsOf: samples)
                }

                return allSamples
        }

        private func makeSingleSppSampler() -> Sampler {
                return .random(RandomSampler(numberOfSamples: 1))
        }

        private func buildImage(from samples: [Sample], resolution: Point2i) -> Image {
                var image = Image(resolution: resolution)
                let iso = camera.film.iso
                for sample in samples {
                        let scaledLight = sample.light * RgbSpectrum(intensity: iso / 100.0)
                        image.addPixel(
                                withColor: scaledLight,
                                withWeight: sample.weight,
                                atLocation: sample.pixel)
                }
                try? image.normalize()
                return image
        }

        private func pushToViewer(
                viewer: ViewerHandle, image: Image, width: Int, height: Int
        ) {
                var floatPixels = [Float](repeating: 0, count: width * height * 3)
                for y in 0..<height {
                        for x in 0..<width {
                                let pixel = image.getPixel(
                                        atLocation: Point2i(x: x, y: y))
                                let idx = (y * width + x) * 3
                                let rgb = pixel.light
                                floatPixels[idx + 0] = Float(rgb.red)
                                floatPixels[idx + 1] = Float(rgb.green)
                                floatPixels[idx + 2] = Float(rgb.blue)
                        }
                }
                viewer_update_framebuffer(
                        viewer, floatPixels, Int32(width), Int32(height))
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
                                let tileBounds = Bounds2i(pMin: pMin, pMax: pMax)
                                tiles.append(Tile(
                                        integrator: integrator,
                                        bounds: tileBounds))
                                minX += tileSize.0
                        }
                        minY += tileSize.1
                }
                return tiles
        }
}
