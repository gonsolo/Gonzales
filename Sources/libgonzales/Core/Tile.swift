import Foundation
import mojoKernel

struct RenderStats: Sendable {
        var bvhTime: TimeInterval = 0
        var shadeTime: TimeInterval = 0
}

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
        ) throws -> (samples: [Sample], stats: RenderStats) {

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

                // Phase 2: Run all bounces
                var samples = [Sample]()
                samples.reserveCapacity(activePaths.count)
                var stats = RenderStats()

                let useGPU = integrator.accelerator.gpuSceneHandle != nil

                if useGPU {
                        // GPU path: keep existing per-bounce loop
                        for _ in 0...integrator.maxDepth {
                                guard !activePaths.isEmpty else { break }

                                var nextActive = [ActivePath]()
                                nextActive.reserveCapacity(activePaths.count)

                                var rays = [Ray]()
                                var tHits = [Real]()
                                rays.reserveCapacity(activePaths.count)
                                tHits.reserveCapacity(activePaths.count)

                                for path in activePaths {
                                        rays.append(path.state.ray)
                                        tHits.append(path.state.tHit)
                                }

                                let bvhStart = Date()
                                let intersectionsC = integrator.accelerator.intersectBatchGPU(
                                        scene: integrator.scene, rays: rays, tHits: &tHits)!
                                stats.bvhTime += Date().timeIntervalSince(bvhStart)

                                let shadeStart = Date()
                                var pathStatesC = [PathState_C]()
                                pathStatesC.reserveCapacity(activePaths.count)
                                for i in 0..<activePaths.count {
                                        let path = activePaths[i]
                                        let r = path.state.ray
                                        let thru = path.state.throughput
                                        let est = path.state.estimate
                                        let alb = path.state.albedo
                                        let rayC = Ray_C(
                                                orgX: Float(r.origin.x), orgY: Float(r.origin.y),
                                                orgZ: Float(r.origin.z),
                                                dirX: Float(r.direction.x), dirY: Float(r.direction.y),
                                                dirZ: Float(r.direction.z))
                                        let pcg1 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                        let pcg2 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                        pathStatesC.append(PathState_C(
                                                ray: rayC,
                                                throughput: (Float(thru.red), Float(thru.green), Float(thru.blue)),
                                                estimate: (Float(est.red), Float(est.green), Float(est.blue)),
                                                albedo: (Float(alb.red), Float(alb.green), Float(alb.blue)),
                                                pcgState: pcg1, pcgInc: pcg2, active: 1))
                                }
                                let handle = integrator.accelerator.gpuSceneHandle!
                                pathStatesC.withUnsafeMutableBufferPointer { pathsPtr in
                                        intersectionsC.withUnsafeBufferPointer { interPtr in
                                                mojo_gpu_shade_batch(
                                                        handle, pathsPtr.baseAddress!,
                                                        Int64(activePaths.count), interPtr.baseAddress!)
                                        }
                                }
                                for i in 0..<activePaths.count {
                                        let pC = pathStatesC[i]
                                        var path = activePaths[i]
                                        if pC.active == 0 {
                                                path.state.estimate = RgbSpectrum(
                                                        red: Real(pC.estimate.0), green: Real(pC.estimate.1),
                                                        blue: Real(pC.estimate.2))
                                                path.state.albedo = RgbSpectrum(
                                                        red: Real(pC.albedo.0), green: Real(pC.albedo.1),
                                                        blue: Real(pC.albedo.2))
                                                samples.append(Sample(
                                                        light: path.state.estimate,
                                                        albedo: path.state.albedo,
                                                        normal: path.state.firstNormal,
                                                        weight: path.state.filterWeight,
                                                        pixel: path.state.pixel))
                                        } else {
                                                path.state.ray = Ray(
                                                        origin: Point(x: Real(pC.ray.orgX), y: Real(pC.ray.orgY), z: Real(pC.ray.orgZ)),
                                                        direction: Vector(x: Real(pC.ray.dirX), y: Real(pC.ray.dirY), z: Real(pC.ray.dirZ)),
                                                        cameraSample: path.state.ray.cameraSample)
                                                path.state.throughput = RgbSpectrum(
                                                        red: Real(pC.throughput.0), green: Real(pC.throughput.1),
                                                        blue: Real(pC.throughput.2))
                                                path.state.albedo = RgbSpectrum(
                                                        red: Real(pC.albedo.0), green: Real(pC.albedo.1),
                                                        blue: Real(pC.albedo.2))
                                                nextActive.append(path)
                                        }
                                }
                                stats.shadeTime += Date().timeIntervalSince(shadeStart)
                                activePaths = nextActive
                        }
                        for path in activePaths {
                                samples.append(Sample(
                                        light: path.state.estimate, albedo: path.state.albedo,
                                        normal: path.state.firstNormal, weight: path.state.filterWeight,
                                        pixel: path.state.pixel))
                        }
                } else {
                        // CPU path: build path states once, run all bounces in Mojo
                        var pathStatesC = [PathState_C]()
                        pathStatesC.reserveCapacity(activePaths.count)
                        for path in activePaths {
                                let r = path.state.ray
                                let rayC = Ray_C(
                                        orgX: Float(r.origin.x), orgY: Float(r.origin.y),
                                        orgZ: Float(r.origin.z),
                                        dirX: Float(r.direction.x), dirY: Float(r.direction.y),
                                        dirZ: Float(r.direction.z))
                                let pcg1 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                let pcg2 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                pathStatesC.append(PathState_C(
                                        ray: rayC,
                                        throughput: (1.0, 1.0, 1.0),
                                        estimate: (0.0, 0.0, 0.0),
                                        albedo: (0.0, 0.0, 0.0),
                                        pcgState: pcg1, pcgInc: pcg2, active: 1))
                        }

                        let shadeStart = Date()
                        integrator.accelerator.renderPaths(
                                scene: integrator.scene,
                                pathStatesC: &pathStatesC,
                                maxDepth: integrator.maxDepth)
                        stats.shadeTime += Date().timeIntervalSince(shadeStart)

                        for i in 0..<activePaths.count {
                                let pC = pathStatesC[i]
                                let path = activePaths[i]
                                samples.append(Sample(
                                        light: RgbSpectrum(
                                                red: Real(pC.estimate.0), green: Real(pC.estimate.1),
                                                blue: Real(pC.estimate.2)),
                                        albedo: RgbSpectrum(
                                                red: Real(pC.albedo.0), green: Real(pC.albedo.1),
                                                blue: Real(pC.albedo.2)),
                                        normal: path.state.firstNormal,
                                        weight: path.state.filterWeight,
                                        pixel: path.state.pixel))
                        }
                }

                return (samples, stats)
        }

        var integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
