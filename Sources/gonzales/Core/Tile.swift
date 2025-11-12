struct Tile: Sendable {

        init(integrator: VolumePathIntegrator, bounds: Bounds2i) {
                self.integrator = integrator
                self.bounds = bounds
        }

        mutating func render(
                reporter: ProgressReporter,
                sampler: inout RandomSampler,
                camera: any Camera,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws -> [Sample] {
                var samples = [Sample]()
                for pixel in bounds {
                        for _ in 0..<sampler.samplesPerPixel {
                                let cameraSample = sampler.getCameraSample(
                                        pixel: pixel, filter: camera.film.filter)
                                let ray = camera.generateRay(cameraSample: cameraSample)
                                var tHit: FloatX = Float.infinity
                                let radianceAlbedoNormal = try integrator.getRadianceAndAlbedo(
                                        from: ray,
                                        tHit: &tHit,
                                        with: &sampler,
                                        lightSampler: &lightSampler,
                                        state: state)

                                let radiance = radianceAlbedoNormal.0
                                let albedo = radianceAlbedoNormal.1
                                let normal = radianceAlbedoNormal.2

                                let dx = cameraSample.film.0 - (FloatX(pixel.x) + 0.5)
                                let dy = cameraSample.film.1 - (FloatX(pixel.y) + 0.5)
                                let filterLocation = Point2f(x: dx, y: dy)
                                let filterValue = camera.film.filter.evaluate(atLocation: filterLocation)
                                let rayWeight = filterValue / cameraSample.filterWeight

                                let sample = Sample(
                                        light: radiance,
                                        albedo: albedo,
                                        normal: normal,
                                        weight: rayWeight,
                                        pixel: pixel)
                                samples.append(sample)

                        }
                }
                return samples
        }

        var integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
