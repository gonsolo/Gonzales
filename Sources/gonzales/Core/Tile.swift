final class Tile: Sendable {

        init(integrator: VolumePathIntegrator, bounds: Bounds2i) {
                self.integrator = integrator
                self.bounds = bounds
        }

        func render(
                reporter: ProgressReporter,
                sampler: RandomSampler,
                camera: any Camera,
                scene: Scene,
                lightSampler: LightSampler,
                state: ImmutableState
        ) throws -> [Sample] {
                var samples = [Sample]()
                //var cameraSamples = [CameraSample]()
                //var rays = [Ray]()
                //var tHits = [Float]()
                for pixel in bounds {
                        for _ in 0..<sampler.samplesPerPixel {
                                let cameraSample = sampler.getCameraSample(pixel: pixel)
                                //cameraSamples.append(cameraSample)
                                let ray = camera.generateRay(cameraSample: cameraSample)
                                //rays.append(ray)
                                //tHits.append(Float.infinity)



                                var tHit: FloatX = Float.infinity
                                let radianceAlbedoNormal = try integrator.getRadiancesAndAlbedos(
                                        from: ray,
                                        tHits: &tHit,
                                        with: sampler,
                                        lightSampler: lightSampler,
                                        state: state)

                        let radiance = radianceAlbedoNormal[0].0
                        let albedo = radianceAlbedoNormal[0].1
                        let normal = radianceAlbedoNormal[0].2
                        let rayWeight: FloatX = 1.0
                   let sample = Sample(
                                light: radiance,
                                albedo: albedo,
                                normal: normal,
                                weight: rayWeight,
                                location: Point2f(x: cameraSample.film.0, y: cameraSample.film.1))
                        samples.append(sample)


                        }
                }
                //let radianceAlbedoNormals = try integrator.getRadiancesAndAlbedos(
                //        from: rays,
                //        tHits: &tHits,
                //        with: sampler,
                //        lightSampler: lightSampler,
                //        state: state)
                //let rayWeight: FloatX = 1.0
                //for (radianceAlbedoNormal, cameraSample) in zip(radianceAlbedoNormals, cameraSamples) {
                //        let radiance = radianceAlbedoNormal.0
                //        let albedo = radianceAlbedoNormal.1
                //        let normal = radianceAlbedoNormal.2
                //        let sample = Sample(
                //                light: radiance,
                //                albedo: albedo,
                //                normal: normal,
                //                weight: rayWeight,
                //                location: Point2f(x: cameraSample.film.0, y: cameraSample.film.1))
                //        samples.append(sample)
                //}
                //await reporter.update()
                return samples
        }

        let integrator: VolumePathIntegrator
        let bounds: Bounds2i
}
