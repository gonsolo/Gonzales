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

                // Phase 2: Process bounces — all paths at bounce N, then bounce N+1
                var samples = [Sample]()
                samples.reserveCapacity(activePaths.count)
                var stats = RenderStats()

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
                        let useGPU = integrator.accelerator.gpuSceneHandle != nil
                        let intersectionsC: [Intersection_C]
                        if useGPU {
                                intersectionsC = integrator.accelerator.intersectBatchGPU(
                                        scene: integrator.scene,
                                        rays: rays,
                                        tHits: &tHits
                                )!
                        } else {
                                intersectionsC = integrator.accelerator.intersectBatchCPU(
                                        scene: integrator.scene,
                                        rays: rays,
                                        tHits: &tHits
                                )
                        }
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
                                    orgX: Float(r.origin.x), orgY: Float(r.origin.y), orgZ: Float(r.origin.z),
                                    dirX: Float(r.direction.x), dirY: Float(r.direction.y), dirZ: Float(r.direction.z)
                                )
                                
                                let thruTup = (Float(thru.red), Float(thru.green), Float(thru.blue))
                                let estTup = (Float(est.red), Float(est.green), Float(est.blue))
                                let albTup = (Float(alb.red), Float(alb.green), Float(alb.blue))
                                let pcg1 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                let pcg2 = UInt64.random(in: 0...UInt64.max, using: &integrator.xoshiro)
                                
                                pathStatesC.append(PathState_C(
                                    ray: rayC,
                                    throughput: thruTup,
                                    estimate: estTup,
                                    albedo: albTup,
                                    pcgState: pcg1,
                                    pcgInc: pcg2,
                                    active: 1
                                ))
                        }
                        
                        if useGPU {
                                let handle = integrator.accelerator.gpuSceneHandle!
                                pathStatesC.withUnsafeMutableBufferPointer { pathsPtr in
                                        intersectionsC.withUnsafeBufferPointer { interPtr in
                                                mojo_gpu_shade_batch(
                                                        handle,
                                                        pathsPtr.baseAddress!,
                                                        Int64(activePaths.count),
                                                        interPtr.baseAddress!
                                                )
                                        }
                                }
                        } else {
                                integrator.accelerator.cpuShadeBatch(
                                        scene: integrator.scene,
                                        pathStatesC: &pathStatesC,
                                        intersectionsC: intersectionsC
                                )
                        }
                        
                        for i in 0..<activePaths.count {
                                let pC = pathStatesC[i]
                                var path = activePaths[i]
                                
                                if pC.active == 0 {
                                        let estR = Real(pC.estimate.0)
                                        let estG = Real(pC.estimate.1)
                                        let estB = Real(pC.estimate.2)
                                        path.state.estimate = RgbSpectrum(red: estR, green: estG, blue: estB)
                                        
                                        let albR = Real(pC.albedo.0)
                                        let albG = Real(pC.albedo.1)
                                        let albB = Real(pC.albedo.2)
                                        path.state.albedo = RgbSpectrum(red: albR, green: albG, blue: albB)
                                        
                                        samples.append(
                                                Sample(
                                                        light: path.state.estimate,
                                                        albedo: path.state.albedo,
                                                        normal: path.state.firstNormal,
                                                        weight: path.state.filterWeight,
                                                        pixel: path.state.pixel))
                                } else {
                                        let orgX = Real(pC.ray.orgX)
                                        let orgY = Real(pC.ray.orgY)
                                        let orgZ = Real(pC.ray.orgZ)
                                        let dirX = Real(pC.ray.dirX)
                                        let dirY = Real(pC.ray.dirY)
                                        let dirZ = Real(pC.ray.dirZ)
                                        let newOrigin = Point(x: orgX, y: orgY, z: orgZ)
                                        let newDirection = Vector(x: dirX, y: dirY, z: dirZ)
                                        path.state.ray = Ray(origin: newOrigin, direction: newDirection, cameraSample: path.state.ray.cameraSample)
                                        
                                        let thruR = Real(pC.throughput.0)
                                        let thruG = Real(pC.throughput.1)
                                        let thruB = Real(pC.throughput.2)
                                        path.state.throughput = RgbSpectrum(red: thruR, green: thruG, blue: thruB)
                                        
                                        let albR = Real(pC.albedo.0)
                                        let albG = Real(pC.albedo.1)
                                        let albB = Real(pC.albedo.2)
                                        path.state.albedo = RgbSpectrum(red: albR, green: albG, blue: albB)
                                        
                                        nextActive.append(path)
                                }
                        }
                        
                        stats.shadeTime += Date().timeIntervalSince(shadeStart)
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

                return (samples, stats)
        }

        var integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
