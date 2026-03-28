struct Tile: Sendable {

        /// An active path being traced through the scene.
        /// Bundles the path state with its per-path sampler state.
        private struct ActivePath {
                var state: PathState
                var sampler: Sampler
                var lightSampler: LightSampler
        }

        mutating func render(
                sampler: inout Sampler,
                camera: any Camera,
                lightSampler: inout LightSampler,
                state: ImmutableState
        ) throws -> [Sample] {

                // Phase 1: Generate all primary rays
                var activePaths = [ActivePath]()
                for pixel in bounds {
                        for sampleIndex in 0..<sampler.samplesPerPixel {
                                var pathSampler = sampler
                                pathSampler.startPixelSample(pixel: pixel, index: sampleIndex)
                                let cameraSample = pathSampler.getCameraSample(
                                        pixel: pixel, filter: camera.film.filter)
                                let ray = camera.generateRay(cameraSample: cameraSample)

                                let deltaX = cameraSample.film.0 - (Real(pixel.x) + 0.5)
                                let deltaY = cameraSample.film.1 - (Real(pixel.y) + 0.5)
                                let filterLocation = Point2f(x: deltaX, y: deltaY)
                                let filterValue = camera.film.filter.evaluate(atLocation: filterLocation)
                                let rayWeight = filterValue / cameraSample.filterWeight

                                let pathState = PathState(
                                        ray: ray,
                                        tHit: Real.infinity,
                                        bounce: 0,
                                        estimate: black,
                                        throughput: white,
                                        albedo: black,
                                        firstNormal: zeroNormal,
                                        pixel: pixel,
                                        filterWeight: rayWeight)

                                activePaths.append(
                                        ActivePath(
                                                state: pathState,
                                                sampler: pathSampler,
                                                lightSampler: lightSampler))
                        }
                }

                // Phase 2: Process bounces — all paths at bounce N, then bounce N+1
                var samples = [Sample]()
                samples.reserveCapacity(activePaths.count)

                for bounce in 0...integrator.maxDepth {
                        guard !activePaths.isEmpty else { break }

                        var nextActive = [ActivePath]()
                        nextActive.reserveCapacity(activePaths.count)

                        for var path in activePaths {
                                path.state.bounce = bounce
                                let continues = try integrator.oneBounce(
                                        state: &path.state,
                                        sampler: &path.sampler,
                                        lightSampler: &path.lightSampler,
                                        immutableState: state)
                                if continues {
                                        nextActive.append(path)
                                } else {
                                        samples.append(
                                                Sample(
                                                        light: path.state.estimate,
                                                        albedo: path.state.albedo,
                                                        normal: path.state.firstNormal,
                                                        weight: path.state.filterWeight,
                                                        pixel: path.state.pixel))
                                }
                        }
                        activePaths = nextActive
                }

                // Any paths that survived all bounces
                for path in activePaths {
                        samples.append(
                                Sample(
                                        light: path.state.estimate,
                                        albedo: path.state.albedo,
                                        normal: path.state.firstNormal,
                                        weight: path.state.filterWeight,
                                        pixel: path.state.pixel))
                }

                return samples
        }

        var integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
