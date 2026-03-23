class RenderConfiguration {

        enum OptionError: Error {
                case camera, crop
        }

        var acceleratorParameters = ParameterDictionary()
        var cameraName = "perspective"
        var cameraParameters = ParameterDictionary()
        var cameraToWorld = Transform()
        var filmName = "rgb"
        var filmParameters = ParameterDictionary()
        var integratorName = "path"
        var integratorParameters = ParameterDictionary()
        var samplerName = "random"
        var samplerParameters = ParameterDictionary()
        var filterName = "gaussian"
        var filterParameters = ParameterDictionary()
        var lights = [Light]()
        var primitives = [IntersectablePrimitive]()
        var objects = [String: [IntersectablePrimitive]]()

        func makeFilm(filter: any Filter, quick: Bool) throws -> Film {
                var x = try filmParameters.findOneInt(called: "xresolution", else: 32)
                var y = try filmParameters.findOneInt(called: "yresolution", else: 32)
                if quick {
                        x /= 4
                        y /= 4
                }
                let resolution = Point2i(x: x, y: y)
                let fileName = try filmParameters.findString(called: "filename") ?? "gonzales.exr"
                let crop = try filmParameters.findReals(called: "cropwindow")
                var cropWindow = Bounds2f()
                if !crop.isEmpty {
                        guard crop.count == 4 else { throw OptionError.crop }
                        cropWindow.pMin = Point2f(x: crop[0], y: crop[2])
                        cropWindow.pMax = Point2f(x: crop[1], y: crop[3])
                }
                let isoValues = try filmParameters.findReals(called: "iso")
                let iso = isoValues.isEmpty ? 100.0 : isoValues[0]
                let maxComponentValues = try filmParameters.findReals(called: "maxcomponentvalue")
                let maxComponentValue = maxComponentValues.isEmpty ? 0.0 : maxComponentValues[0]
                return Film(
                        name: fileName,
                        resolution: resolution,
                        fileName: fileName,
                        filter: filter,
                        crop: cropWindow,
                        iso: iso,
                        maxComponentValue: maxComponentValue
                )
        }

        func makeFilter(name: String, parameters: ParameterDictionary) throws -> any Filter {

                func makeSupport(withDefault support: (Real, Real) = (2, 2)) throws -> Vector2F {
                        let xwidth = try parameters.findOneReal(called: "xradius", else: support.0)
                        let ywidth = try parameters.findOneReal(called: "yradius", else: support.1)
                        return Vector2F(x: xwidth, y: ywidth)
                }

                var filter: any Filter
                switch name {
                case "box":
                        let support = try makeSupport(withDefault: (0.5, 0.5))
                        filter = BoxFilter(support: support)
                case "gaussian":
                        let support = try makeSupport(withDefault: (1.5, 1.5))
                        let sigma = try parameters.findOneReal(called: "sigma", else: 0.5)
                        filter = GaussianFilter(withSupport: support, withSigma: sigma)
                // mitchell missing
                // sinc missing
                case "triangle":
                        let support = try makeSupport(withDefault: (2, 2))
                        filter = TriangleFilter(support: support)
                default:
                        throw RenderError.unimplemented(function: #function, file: #filePath, line: #line, message: "Unknown pixel filter: \(name)")
                }
                return filter
        }

        func makeCamera(quick: Bool) throws -> PerspectiveCamera {
                guard cameraName == "perspective" else { throw OptionError.camera }
                let filter = try makeFilter(name: filterName, parameters: filterParameters)
                let film = try makeFilm(filter: filter, quick: quick)
                let resolution = film.getResolution()
                let frame = Real(resolution.x) / Real(resolution.y)
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
                let screenWindow = try cameraParameters.findReals(called: "screenwindow")
                if !screenWindow.isEmpty {
                        if screenWindow.count != 4 {
                                print("Warning: screenwindow must be four coordinates!")
                        } else {
                                screen.pMin.x = screenWindow[0]
                                screen.pMax.x = screenWindow[1]
                                screen.pMin.y = screenWindow[2]
                                screen.pMax.y = screenWindow[3]
                        }
                }
                let fov = try cameraParameters.findOneReal(called: "fov", else: 30)
                let focalDistance = try cameraParameters.findOneReal(
                        called: "focaldistance",
                        else: 1e6)
                let lensRadius = try cameraParameters.findOneReal(called: "lensradius", else: 0)
                return try PerspectiveCamera(
                        cameraToWorld: cameraToWorld,
                        screenWindow: screen,
                        fov: fov,
                        focalDistance: focalDistance,
                        lensRadius: lensRadius,
                        film: film
                )
        }

        func makeIntegrator(sampler _: Sampler, accelerator: Accelerator, scene: Scene) throws
                -> VolumePathIntegrator {
                switch self.integratorName {
                case "path": break
                case "volpath": break
                default:
                        var message = "Integrator \(self.integratorName) not implemented, "
                        message += "using path integrator!"
                        print("Warning: \(message)")
                }
                let maxDepth = try self.integratorParameters.findOneInt(
                        called: "maxdepth",
                        else: 5
                )
                return VolumePathIntegrator(maxDepth: maxDepth, accelerator: accelerator, scene: scene)
        }

        func makeSampler(film: Film, quick: Bool) throws -> Sampler {
                switch samplerName {
                case "sobol", "zsobol":
                        return try .sobol(
                                createZSobolSampler(
                                        parameters: samplerParameters, fullResolution: film.resolution,
                                        quick: quick))
                case "random":
                        return try .random(
                                createRandomSampler(
                                        parameters: samplerParameters, quick: quick))
                default:
                        print("Warning: Unknown sampler, using random sampler.")
                        return try .random(
                                createRandomSampler(
                                        parameters: samplerParameters, quick: quick))
                }
        }

        private func cleanUp() {
                primitives.removeAll()
                objects.removeAll()
        }

        // swiftlint:disable:next function_parameter_count
        func makeRenderer(
                geometricPrimitives: [GeometricPrimitive],
                areaLights: [AreaLight],
                materials: [Material],
                transformedPrimitives: [TransformedPrimitive],
                acceleratorName: String,
                immutableState: ImmutableState,
                renderOptions: RenderOptions,
                meshes: TriangleMeshes,
                arena: TextureArena
        ) async throws -> some Renderer {
                let camera = try makeCamera(quick: renderOptions.quick)
                let sampler = try makeSampler(film: camera.film, quick: renderOptions.quick)
                let scene = Scene(
                        lights: lights,
                        materials: materials,
                        meshes: meshes,
                        geometricPrimitives: geometricPrimitives,
                        areaLights: areaLights,
                        transformedPrimitives: transformedPrimitives,
                        arena: arena)

                let accelerator = try await makeAccelerator(
                        scene: scene, primitives: primitives,
                        acceleratorName: acceleratorName)
                
                cleanUp()

                let integrator = try makeIntegrator(sampler: sampler, accelerator: accelerator, scene: scene)
                let powerLightSampler = try PowerLightSampler(
                        sampler: sampler, lights: lights, scene: scene)
                let lightSampler = LightSampler(powerLightSampler: powerLightSampler)
                let tileSize = (32, 32)

                return TileRenderer(
                        camera: camera,
                        integrator: integrator,
                        sampler: sampler,
                        lightSampler: lightSampler,
                        tileSize: tileSize,
                        immutableState: immutableState,
                        renderOptions: renderOptions
                )
        }
}
