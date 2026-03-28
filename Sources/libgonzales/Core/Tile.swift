struct Tile: Sendable {

        mutating func render(
                sampler: inout Sampler,
                camera: any Camera,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws -> [Sample] {
                var samples = [Sample]()
                for pixel in bounds {
                        for sampleIndex in 0..<sampler.samplesPerPixel {
                                sampler.startPixelSample(pixel: pixel, index: sampleIndex)
                                let cameraSample = sampler.getCameraSample(
                                        pixel: pixel, filter: camera.film.filter)
                                let ray = camera.generateRay(cameraSample: cameraSample)

                                let deltaX = cameraSample.film.0 - (Real(pixel.x) + 0.5)
                                let deltaY = cameraSample.film.1 - (Real(pixel.y) + 0.5)
                                let filterLocation = Point2f(x: deltaX, y: deltaY)
                                let filterValue = camera.film.filter.evaluate(atLocation: filterLocation)
                                let rayWeight = filterValue / cameraSample.filterWeight

                                var pathState = PathState(
                                        ray: ray,
                                        tHit: Real.infinity,
                                        bounce: 0,
                                        estimate: black,
                                        throughput: white,
                                        albedo: black,
                                        firstNormal: zeroNormal,
                                        pixel: pixel,
                                        filterWeight: rayWeight)

                                try integrator.evaluateRayPath(
                                        state: &pathState,
                                        with: &sampler,
                                        lightSampler: &lightSampler,
                                        immutableState: state)

                                let sample = Sample(
                                        light: pathState.estimate,
                                        albedo: pathState.albedo,
                                        normal: pathState.firstNormal,
                                        weight: pathState.filterWeight,
                                        pixel: pathState.pixel)
                                samples.append(sample)
                        }
                }
                return samples
        }

        var integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
