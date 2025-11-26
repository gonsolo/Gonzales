struct Tile: Sendable {

        mutating func render(
                sampler: inout Sampler,
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
                                let rayTraceSample = try integrator.evaluateRayPath(
                                        from: ray,
                                        tHit: &tHit,
                                        with: &sampler,
                                        lightSampler: &lightSampler,
                                        state: state)

                                let dx = cameraSample.film.0 - (FloatX(pixel.x) + 0.5)
                                let dy = cameraSample.film.1 - (FloatX(pixel.y) + 0.5)
                                let filterLocation = Point2f(x: dx, y: dy)
                                let filterValue = camera.film.filter.evaluate(atLocation: filterLocation)
                                let rayWeight = filterValue / cameraSample.filterWeight

                                let sample = Sample(
                                        light: rayTraceSample.estimate,
                                        albedo: rayTraceSample.albedo,
                                        normal: rayTraceSample.normal,
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
