struct Tile {

        func render(
                reporter: ProgressReporter,
                scene: Scene,
                sampler: Sampler,
                camera: Camera,
                integrator: Integrator
        ) throws -> [Sample] {
                var samples = [Sample]()
                for pixel in bounds {
                        let pixelSamples = try renderPixel(
                                pixel: pixel,
                                reporter: reporter,
                                scene: scene,
                                sampler: sampler,
                                camera: camera,
                                integrator: integrator)
                        samples.append(contentsOf: pixelSamples)
                }
                return samples
        }

        func renderPixel(
                pixel: Point2I,
                reporter: ProgressReporter,
                scene: Scene,
                sampler: Sampler,
                camera: Camera,
                integrator: Integrator
        ) throws -> [Sample] {
                var samples = [Sample]()
                for _ in 0..<sampler.samplesPerPixel {
                        let cameraSample = sampler.getCameraSample(pixel: pixel)
                        let ray = camera.generateRay(sample: cameraSample)
                        var tHit = Float.infinity
                        let (L, albedo, normal) = try integrator.getRadianceAndAlbedo(
                                from: ray,
                                tHit: &tHit,
                                for: scene,
                                with: sampler)
                        let rayWeight: FloatX = 1.0
                        let sample = Sample(
                                light: L,
                                albedo: albedo,
                                normal: normal,
                                weight: rayWeight,
                                location: cameraSample.film)
                        samples.append(sample)
                        reporter.update()
                }
                return samples
        }

        static let size = 64
        var bounds: Bounds2i
}
