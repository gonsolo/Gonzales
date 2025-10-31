class Options {

        enum OptionError: Error {
                case camera, crop
        }

        var acceleratorParameters = ParameterDictionary()
        var cameraName = "perspective"
        var cameraParameters = ParameterDictionary()
        var cameraToWorld = Transform()
        var filmName = "image"
        var filmParameters = ParameterDictionary()
        var integratorName = "path"
        var integratorParameters = ParameterDictionary()
        var samplerName = "random"
        var samplerParameters = ParameterDictionary()
        var filterName = "gaussian"
        var filterParameters = ParameterDictionary()
        var lights = [Light]()
        var primitives = [any Boundable & Intersectable]()
        var objects = ["": [any Boundable & Intersectable]()]

        var defaultMaterial: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))

        @MainActor
        init() {
                materials.append(defaultMaterial)
        }

        @MainActor
        func makeFilm(filter: any Filter) throws -> Film {
                var x = try filmParameters.findOneInt(called: "xresolution", else: 32)
                var y = try filmParameters.findOneInt(called: "yresolution", else: 32)
                if quick {
                        x /= 4
                        y /= 4
                }
                let resolution = Point2i(x: x, y: y)
                let fileName = try filmParameters.findString(called: "filename") ?? "gonzales.exr"
                let crop = try filmParameters.findFloatXs(called: "cropwindow")
                var cropWindow = Bounds2f()
                if !crop.isEmpty {
                        guard crop.count == 4 else { throw OptionError.crop }
                        cropWindow.pMin = Point2f(x: crop[0], y: crop[2])
                        cropWindow.pMax = Point2f(x: crop[1], y: crop[3])
                }
                return Film(
                        name: fileName,
                        resolution: resolution,
                        fileName: fileName,
                        filter: filter,
                        crop: cropWindow
                )
        }

        func makeFilter(name: String, parameters: ParameterDictionary) throws -> any Filter {

                func makeSupport(withDefault support: (FloatX, FloatX) = (2, 2)) throws -> Vector2F {
                        let xwidth = try parameters.findOneFloatX(called: "xradius", else: support.0)
                        let ywidth = try parameters.findOneFloatX(called: "yradius", else: support.1)
                        return Vector2F(x: xwidth, y: ywidth)
                }

                var filter: any Filter
                switch name {
                case "box":
                        let support = try makeSupport(withDefault: (0.5, 0.5))
                        filter = BoxFilter(support: support)
                case "gaussian":
                        let support = try makeSupport(withDefault: (1.5, 1.5))
                        let sigma = try parameters.findOneFloatX(called: "sigma", else: 0.5)
                        filter = GaussianFilter(withSupport: support, withSigma: sigma)
                // mitchell missing
                // sinc missing
                case "triangle":
                        let support = try makeSupport(withDefault: (2, 2))
                        filter = TriangleFilter(support: support)
                default:
                        fatalError("Unknown pixel filter!")
                }
                return filter
        }

        @MainActor
        func makeCamera() async throws -> any Camera {
                guard cameraName == "perspective" else { throw OptionError.camera }
                let filter = try makeFilter(name: filterName, parameters: filterParameters)
                let film = try makeFilm(filter: filter)
                let resolution = await film.getResolution()
                let frame = FloatX(resolution.x) / FloatX(resolution.y)
                var screen = Bounds2f()

                if frame > 1 {
                        // Blender does not like this
                        screen.pMin.x = -frame
                        screen.pMax.x = +frame
                        screen.pMin.y = -1
                        screen.pMax.y = +1
                } else {
                        screen.pMin.x = -1
                        screen.pMax.x = +1
                        screen.pMin.y = -1 / frame
                        screen.pMax.y = +1 / frame
                }
                let screenWindow = try cameraParameters.findFloatXs(called: "screenwindow")
                if !screenWindow.isEmpty {
                        if screenWindow.count != 4 {
                                warning("screenwindow must be four coordinates!")
                        } else {
                                screen.pMin.x = screenWindow[0]
                                screen.pMax.x = screenWindow[1]
                                screen.pMin.y = screenWindow[2]
                                screen.pMax.y = screenWindow[3]
                        }
                }
                let fov = try cameraParameters.findOneFloatX(called: "fov", else: 30)
                let focalDistance = try cameraParameters.findOneFloatX(
                        called: "focaldistance",
                        else: 1e6)
                let lensRadius = try cameraParameters.findOneFloatX(called: "lensradius", else: 0)
                return try await PerspectiveCamera(
                        cameraToWorld: cameraToWorld,
                        screenWindow: screen,
                        fov: fov,
                        focalDistance: focalDistance,
                        lensRadius: lensRadius,
                        film: film
                )
        }

        @MainActor
        func makeIntegrator(sampler: RandomSampler) throws -> VolumePathIntegrator {
                switch options.integratorName {
                case "path": break
                case "volpath": break
                default:
                        var message = "Integrator \(options.integratorName) not implemented, "
                        message += "using path integrator!"
                        warning(message)
                }
                let maxDepth = try options.integratorParameters.findOneInt(
                        called: "maxdepth",
                        else: 1
                )
                return VolumePathIntegrator(maxDepth: maxDepth)
        }

        @MainActor
        func makeSampler(film: Film) throws -> RandomSampler {
                if samplerName != "random" {
                        warning("Unknown sampler, using random sampler.")
                }
                let sampler = try createRandomSampler(parameters: samplerParameters, quick: quick)
                return sampler
        }

        private func cleanUp() {
                primitives.removeAll()
                objects.removeAll()
        }

        @MainActor
        func makeRenderer() async throws -> some Renderer {
                let camera = try await makeCamera()
                let sampler = try makeSampler(film: camera.film)
                scene.addLights(lights: lights)
                scene.addMaterials(materials: materials)
                scene.addMeshes(meshes: triangleMeshBuilder.getMeshes())
                scene.addGeometricPrimivites(geometricPrimitives: geometricPrimitives)
                let acceleratorTimer = Timer("Build accelerator...", newline: false)
                accelerator = try await makeAccelerator(primitives: primitives)
                cleanUp()
                print("Building accelerator: \(acceleratorTimer.elapsed)")
                let integrator = try makeIntegrator(sampler: sampler)
                let powerLightSampler = await PowerLightSampler(sampler: sampler, lights: lights)
                let lightSampler = LightSampler.power(powerLightSampler)
                let tileSize = (32, 32)

                return await TileRenderer(
                        camera: camera,
                        integrator: integrator,
                        sampler: sampler,
                        lightSampler: lightSampler,
                        tileSize: tileSize
                )
        }
}
