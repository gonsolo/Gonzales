import Foundation

func makeAccelerator(
        scene: Scene,
        primitives: [IntersectablePrimitive],
        acceleratorName: String
) async throws -> Accelerator {
        switch acceleratorName {
        case "bvh":
                let builder = try await BoundingHierarchyBuilder(scene: scene, primitives: primitives)
                let boundingHierarchy = try builder.getBoundingHierarchy()
                let accelerator = Accelerator(boundingHierarchy: boundingHierarchy)
                return accelerator
        default:
                throw SceneDescriptionError.accelerator
        }
}

func lookAtTransform(eye: Point, target: Point, upVector: Vector) throws -> Transform {
        let dir: Vector = normalized(target - eye)
        var upVector = normalized(upVector)
        let right = normalized(cross(upVector, dir))
        if length(right) == 0 { return Transform() }
        upVector = cross(dir, right)
        let matrix = Matrix(
                t00: right.x, t01: upVector.x, t02: dir.x, t03: eye.x,
                t10: right.y, t11: upVector.y, t12: dir.y, t13: eye.y,
                t20: right.z, t21: upVector.z, t22: dir.z, t23: eye.z,
                t30: 0, t31: 0, t32: 0, t33: 1
        )
        let transform = try Transform(matrix: matrix.inverse)
        return transform
}

public enum SceneDescriptionError: Error {
        case accelerator
        case areaLight
        case coordSysTransform
        case input(message: String)
        case makeLight(message: String)
        case makeSampler
        case makeShapes(message: String)
        case namedMedium
        case namedMaterial
        case objectInstance
        case parseAttributeEnd
        case ply(message: String)
        case unknownTextureFormat(suffix: String)
        case transformsEmpty
        case unknownAccelerator
        case unknownIntegrator
        case unknownTexture(name: String)
        case writeUse
        case wrongType(message: String)
}

public class SceneDescription {

        final class InstanceCreationDesc {
                let name: String
                let currentTransform: Transform

                init(name: String, currentTransform: Transform) {
                        self.name = name
                        self.currentTransform = currentTransform
                }
        }
        private var uninstantiatedInstances = [InstanceCreationDesc]()
        var transformedPrimitives = [TransformedPrimitive]()

        var apiGeometricPrimitives = [GeometricPrimitive]()
        var areaLights = [AreaLight]()

        struct DeferredShapeBatch {
                let job: @Sendable () throws -> [ShapeType]
                let isAreaLight: Bool
                let areaLightName: String
                let areaLightParameters: ParameterDictionary
                let alpha: Real
                let reverseOrientation: Bool
                let currentMediumInterface: MediumInterface?
                let objectName: String?
                let materialIndex: Int
        }
        var shapeBatches = [DeferredShapeBatch]()

        var acceleratorName = "bvh"
        var currentTransform = Transform()
        var materials = [Material]()
        var namedCoordinateSystems = [String: Transform]()
        var options: RenderConfiguration
        var readReporter: ProgressReporter?
        var readProgressTask: Task<Void, Never>?
        var renderOptions: RenderOptions
        var state: State
        var states = [State]()
        var transforms = [Transform]()
        var triangleMeshBuilder = TriangleMeshBuilder()

        public init(renderOptions: RenderOptions) {
                self.options = RenderConfiguration()
                self.renderOptions = renderOptions
                self.state = State(ptexMemory: renderOptions.ptexMemory)
                let idx = self.state.arena.appendRgb(
                        RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))
                self.materials.append(Material.diffuse(Diffuse(reflectance: Texture.rgbSpectrumTexture(idx))))
        }
}

extension SceneDescription {

        func attributeBegin() throws {
                try transformBegin()
                states.append(state)
        }

        func attributeEnd() throws {
                try transformEnd()
                guard var last = states.popLast() else {
                        throw SceneDescriptionError.parseAttributeEnd
                }
                // Preserve the current arena — textures added inside this scope
                // must remain valid since materials may reference their indices.
                last.arena = state.arena
                state = last
        }

        func camera(name _: String, parameters: ParameterDictionary) throws {
                options.cameraName = "perspective"
                options.cameraParameters = parameters
                options.cameraToWorld = currentTransform.inverse
                namedCoordinateSystems["camera"] = options.cameraToWorld
        }

        func coordinateSystem(name: String) {
                namedCoordinateSystems[name] = currentTransform
        }

        func coordSysTransform(name: String) throws {
                guard let transform = namedCoordinateSystems[name] else {
                        throw SceneDescriptionError.coordSysTransform
                }
                currentTransform = transform
        }

        func concatTransform(values: [Real]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                try (currentTransform *= Transform(matrix: matrix))
        }

        func film(name: String, parameters: ParameterDictionary) throws {
                let fileName = try parameters.findString(called: "filename") ?? "gonzales.exr"
                guard fileName.hasSuffix("exr") else {
                        throw SceneDescriptionError.input(message: "Only exr output supported!")
                }
                options.filmName = name
                options.filmParameters = parameters
        }

        func identity() {
                currentTransform = Transform()
        }

        func importFile(file sceneName: String) async throws {
                try await include(file: sceneName, render: false)
        }

        public func include(file sceneName: String, render: Bool) async throws {
                // print(sceneName)
                do {
                        let fileManager = FileManager.default
                        let absoluteSceneName = renderOptions.sceneDirectory + "/" + sceneName
                        var components = absoluteSceneName.components(separatedBy: ".")
                        components.removeLast()
                        guard fileManager.fileExists(atPath: absoluteSceneName) else {
                                throw RenderError.fileNotExisting(name: absoluteSceneName)
                        }
                        if #available(OSX 10.15, *) {
                                let parser = try Parser(
                                        fileName: absoluteSceneName,
                                        sceneDescription: self, render: render)
                                try await parser.parse()
                        } else {
                                // Fallback on earlier versions
                        }
                } catch SceneDescriptionError.wrongType(let message) {
                        print("Error: Wrong type: \(message) in \(sceneName).")
                }
        }

        func areaLight(name: String, parameters: ParameterDictionary) throws {
                guard name == "diffuse" || name == "area" else {
                        throw SceneDescriptionError.areaLight
                }
                state.areaLight = name
                state.areaLightParameters = parameters
        }

        func accelerator(name: String, parameters _: ParameterDictionary) throws {
                switch name {
                case "bvh":
                        acceleratorName = "bvh"
                default:
                        throw SceneDescriptionError.accelerator
                }
        }

        func integrator(name: String, parameters: ParameterDictionary) throws {
                options.integratorName = name
                options.integratorParameters = parameters
        }

        func lightSource(name: String, parameters: ParameterDictionary) throws {
                let light = try makeLight(
                        name: name,
                        parameters: parameters,
                        lightToWorld: currentTransform)
                options.lights.append(light)
        }

        func lookAt(eye: Point, target: Point, upVector: Vector) throws {
                let transform = try lookAtTransform(eye: eye, target: target, upVector: upVector)
                try (currentTransform *= transform)
        }

        func makeNamedMaterial(name: String, parameters: ParameterDictionary) throws {
                let type = try parameters.findString(called: "type") ?? "defaultMaterial"
                state.namedMaterials[name] = UninstancedMaterial(type: type, parameters: parameters)
        }

        func makeNamedMedium(name: String, parameters: ParameterDictionary) throws {
                guard let type = try parameters.findString(called: "type") else {
                        throw SceneDescriptionError.namedMedium
                }
                switch type {
                case "cloud":
                        print("Warning: Cloud is not implemented!")
                case "homogeneous":
                        let scale = try parameters.findOneReal(called: "scale", else: 1)
                        let absorption =
                                try parameters.findSpectrum(name: "sigma_a", else: white)?.asRgb() ?? white
                        let scattering =
                                try parameters.findSpectrum(name: "sigma_s", else: white)?.asRgb() ?? white
                        state.namedMedia[name] = Homogeneous(
                                scale: scale,
                                absorption: absorption,
                                scattering: scattering)
                case "nanovdb":
                        print("Warning: Nanovdb is not implemented!")
                case "uniformgrid":
                        print("Warning: Uniform grid is not implemented!")
                default:
                        throw SceneDescriptionError.namedMedium
                }
        }

        func material(type: String, parameters: ParameterDictionary) throws {
                state.currentMaterial = UninstancedMaterial(type: type, parameters: parameters)
        }

        func mediumInterface(interior: String, exterior: String) {
                state.currentMediumInterface = MediumInterface(interior: interior, exterior: exterior)
        }

        func namedMaterial(name: String) throws {
                state.currentNamedMaterial = name
        }

        func objectBegin(name: String) throws {
                try attributeBegin()
                state.objectName = name
        }

        func objectEnd() throws {
                try attributeEnd()
                state.objectName = nil
        }

        func objectInstance(name: String) throws {
                let desc = InstanceCreationDesc(name: name, currentTransform: currentTransform)
                uninstantiatedInstances.append(desc)
        }

        func sampler(name: String, parameters: ParameterDictionary) {
                options.samplerName = name
                options.samplerParameters = parameters
        }

        func pixelFilter(name: String, parameters: ParameterDictionary) {
                options.filterName = name
                options.filterParameters = parameters
        }

        func shape(name: String, parameters: ParameterDictionary) throws {
                let currentTransform = self.currentTransform
                let sceneDirectory = self.renderOptions.sceneDirectory
                let builder = self.triangleMeshBuilder
                let acceleratorNameForShapes = self.acceleratorName

                let alpha = try parameters.findOneReal(called: "alpha", else: 1)
                let material = try state.createMaterial(
                        parameters: parameters,
                        currentMaterial: state.currentMaterial,
                        currentNamedMaterial: state.currentNamedMaterial,
                        textures: state.textures
                )

                let isAreaLight = !state.areaLight.isEmpty
                let areaLightName = state.areaLight
                let areaLightParameters = state.areaLightParameters
                let reverseOrientation = state.reverseOrientation
                let currentMediumInterface = state.currentMediumInterface
                let objectName = state.objectName

                var materialIndex = noMaterial
                if !isAreaLight {
                        materialIndex = materials.count
                        materials.append(material)
                }

                let job: @Sendable () throws -> [ShapeType] = {
                        switch name {
                        case "bilinearmesh":
                                return []  // Ignore for now
                        case "curve":
                                return try Curve.createShape(
                                        objectToWorld: currentTransform,
                                        parameters: parameters,
                                        acceleratorName: acceleratorNameForShapes)
                        case "cylinder":
                                return []  // Ignore for now
                        case "disk":
                                return try Disk.create(
                                        objectToWorld: currentTransform,
                                        parameters: parameters)
                        case "loopsubdiv":
                                return try Triangle.createFromParameters(
                                        objectToWorld: currentTransform,
                                        parameters: parameters,
                                        triangleMeshBuilder: builder)
                        case "plymesh":
                                return try PlyMesh.create(
                                        objectToWorld: currentTransform,
                                        parameters: parameters,
                                        sceneDirectory: sceneDirectory,
                                        triangleMeshBuilder: builder)
                        case "sphere":
                                return [
                                        try Sphere.create(
                                                objectToWorld: currentTransform,
                                                parameters: parameters)
                                ]
                        case "trianglemesh":
                                return try Triangle.createFromParameters(
                                        objectToWorld: currentTransform,
                                        parameters: parameters,
                                        triangleMeshBuilder: builder)
                        default:
                                throw SceneDescriptionError.makeShapes(message: name)
                        }
                }

                let batch = DeferredShapeBatch(
                        job: job,
                        isAreaLight: isAreaLight,
                        areaLightName: areaLightName,
                        areaLightParameters: areaLightParameters,
                        alpha: alpha,
                        reverseOrientation: reverseOrientation,
                        currentMediumInterface: currentMediumInterface,
                        objectName: objectName,
                        materialIndex: materialIndex
                )

                shapeBatches.append(batch)
        }

        private func resolveInstances() async throws {

                // Now resolve Object Instances using a dummy Scene for accelerator building bounds
                let tempScene = Scene(
                        lights: [Light](),
                        materials: materials,
                        meshes: triangleMeshBuilder.getMeshes(),
                        geometricPrimitives: apiGeometricPrimitives,
                        areaLights: areaLights,
                        transformedPrimitives: transformedPrimitives, arena: state.arena)

                var instancedAccelerators = [String: Accelerator]()

                // 1. Identify unique objects that need an accelerator
                var uniqueNamesSet = Set<String>()
                var uniqueNames = [String]()
                for instanceDesc in uninstantiatedInstances where !uniqueNamesSet.contains(instanceDesc.name)
                {
                        uniqueNamesSet.insert(instanceDesc.name)
                        uniqueNames.append(instanceDesc.name)
                }

                // 2. Build them concurrently
                final class LocalInstancedAccelerators: @unchecked Sendable {
                        var results: [Accelerator?]
                        let lock = NSLock()
                        init(count: Int) { self.results = [Accelerator?](repeating: nil, count: count) }
                        func set(index: Int, accelerator: Accelerator) {
                                lock.lock()
                                self.results[index] = accelerator
                                lock.unlock()
                        }
                }

                let localInstancedAccelerators = LocalInstancedAccelerators(count: uniqueNames.count)
                let immutableUniqueNames = uniqueNames
                let localAcceleratorName = self.acceleratorName
                let immutableObjects = options.objects  // Value-type copy for Sendable closure capture

                await withTaskGroup(of: Void.self) { group in
                        for index in 0..<immutableUniqueNames.count {
                                group.addTask {
                                        let name = immutableUniqueNames[index]
                                        guard let prims = immutableObjects[name] else { return }
                                        do {
                                                let accelerator = try await makeAccelerator(
                                                        scene: tempScene,
                                                        primitives: prims,
                                                        acceleratorName: localAcceleratorName)
                                                localInstancedAccelerators.set(
                                                        index: index, accelerator: accelerator)
                                        } catch {
                                                print("Error building instanced local BVH: \(error)")
                                        }
                                }
                        }
                }

                // 3. Populate cache dictionary
                for (index, name) in immutableUniqueNames.enumerated() {
                        if let acc = localInstancedAccelerators.results[index] {
                                instancedAccelerators[name] = acc
                        }
                }

                // 4. Resolve instances sequentially to preserve idx order
                for instanceDesc in uninstantiatedInstances {
                        guard options.objects[instanceDesc.name] != nil else {
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line,
                                        message: "Missing object instance \(instanceDesc.name)")
                        }

                        guard let accelerator = instancedAccelerators[instanceDesc.name] else {
                                continue
                        }

                        let transformedPrimitive = TransformedPrimitive(
                                accelerator: accelerator,
                                transform: instanceDesc.currentTransform,
                                idx: transformedPrimitives.count)
                        options.primitives.append(.transformedPrimitive(transformedPrimitive))
                        transformedPrimitives.append(transformedPrimitive)
                }
        }

        func transform(values: [Real]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform = try Transform(matrix: matrix)
        }

        func transformBegin() throws {
                transforms.append(currentTransform)
        }

        func transformEnd() throws {
                guard let last = transforms.popLast() else {
                        throw SceneDescriptionError.transformsEmpty
                }
                currentTransform = last
        }

        func scale(x: Real, y: Real, z: Real) throws {
                let matrix = Matrix(
                        t00: x, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: y, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: z, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                try (currentTransform *= Transform(matrix: matrix))
        }

        // Not really part of PBRT API

        public func start() {
                let reporter = ProgressReporter(title: "Reading")
                self.readReporter = reporter
                self.readProgressTask = Task { await runProgressReporter(reporter: reporter) }
        }

        func rotate(by angle: Angle, around axis: Vector) throws {
                let normalizedAxis = normalized(axis)
                let theta = radians(deg: angle)
                let sinTheta = sin(theta)
                let cosTheta = cos(theta)
                let t00 =
                        normalizedAxis.x * normalizedAxis.x + (1 - normalizedAxis.x * normalizedAxis.x)
                        * cosTheta
                let t01 = normalizedAxis.x * normalizedAxis.y * (1 - cosTheta) - normalizedAxis.z * sinTheta
                let t02 = normalizedAxis.x * normalizedAxis.z * (1 - cosTheta) + normalizedAxis.y * sinTheta
                let t10 = normalizedAxis.x * normalizedAxis.y * (1 - cosTheta) + normalizedAxis.z * sinTheta
                let t11 =
                        normalizedAxis.y * normalizedAxis.y + (1 - normalizedAxis.y * normalizedAxis.y)
                        * cosTheta
                let t12 = normalizedAxis.y * normalizedAxis.z * (1 - cosTheta) - normalizedAxis.x * sinTheta
                let t20 = normalizedAxis.x * normalizedAxis.z * (1 - cosTheta) - normalizedAxis.y * sinTheta
                let t21 = normalizedAxis.y * normalizedAxis.z * (1 - cosTheta) + normalizedAxis.x * sinTheta
                let t22 =
                        normalizedAxis.z * normalizedAxis.z + (1 - normalizedAxis.z * normalizedAxis.z)
                        * cosTheta
                let matrix = Matrix(
                        t00: t00, t01: t01, t02: t02, t03: 0,
                        t10: t10, t11: t11, t12: t12, t13: 0,
                        t20: t20, t21: t21, t22: t22, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                try (currentTransform *= Transform(matrix: matrix))
        }

        func texture(
                name: String,
                type: String,
                textureClass: String,
                parameters: ParameterDictionary
        )
                throws
        {
                guard type == "spectrum" || type == "float" || type == "color" else {
                        print("Warning: Unimplemented texture type: \(type)")
                        return
                }
                var texture: Texture
                switch textureClass {
                case "checkerboard":
                        throw RenderError.unimplemented(
                                function: #function, file: #filePath, line: #line, message: "")
                case "constant":
                        switch type {
                        case "spectrum", "color":
                                texture = try parameters.findRgbSpectrumTexture(
                                        name: "value", textures: state.textures, arena: &state.arena)
                        case "float":
                                texture = try parameters.findRealTexture(
                                        name: "value", textures: state.textures, arena: &state.arena)
                        default:
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line, message: "")
                        }
                case "imagemap":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(
                                name: fileName, type: type,
                                sceneDirectory: renderOptions.sceneDirectory, arena: &state.arena)
                case "mix":
                        switch type {
                        case "color", "spectrum":
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line, message: "")
                        case "float":
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line, message: "")
                        default:
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line, message: "")
                        }
                case "ptex":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(
                                name: fileName, type: type,
                                sceneDirectory: renderOptions.sceneDirectory, arena: &state.arena)
                case "scale":
                        let scale = try parameters.findRealTexture(
                                name: "scale", textures: state.textures, arena: &state.arena)
                        switch type {
                        case "spectrum", "color":
                                let tex = try parameters.findRgbSpectrumTexture(
                                        name: "tex", textures: state.textures, arena: &state.arena)
                                let scaledTexture = ScaledTextureRgb(
                                        tex: tex.index, scale: scale.index)
                                let idx = state.arena.appendRgb(
                                        RgbSpectrumTexture.scaledTexture(scaledTexture))
                                texture = Texture.rgbSpectrumTexture(idx)
                        case "float":
                                let tex = try parameters.findRealTexture(
                                        name: "tex", textures: state.textures, arena: &state.arena)
                                let scaledTexture = ScaledTextureFloat(
                                        tex: tex.index, scale: scale.index)
                                let idx = state.arena.appendFloat(FloatTexture.scaledTexture(scaledTexture))
                                texture = Texture.floatTexture(idx)
                        default:
                                throw RenderError.unimplemented(
                                        function: #function, file: #filePath, line: #line,
                                        message: "Unknown scale type")
                        }
                default:
                        print("Warning: Unimplemented texture class: \(textureClass)")
                        return
                }
                state.textures[name] = texture
        }

        func translate(amount: Vector) throws {
                let matrix = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: amount.x,
                        t10: 0, t11: 1, t12: 0, t13: amount.y,
                        t20: 0, t21: 0, t22: 1, t23: amount.z,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let translation = try Transform(matrix: matrix)
                try (currentTransform *= translation)
        }

        public func worldBegin() {
                currentTransform = Transform()
        }

        public func reverseOrientation() {
                state.reverseOrientation.toggle()
        }

        func worldEnd() async throws {
                // PASS 1: Execute shape batches concurrently to parse PLY meshes across 12 cores
                final class JobResults: @unchecked Sendable {
                        var results: [[ShapeType]?]
                        let lock = NSLock()
                        init(count: Int) { self.results = [[ShapeType]?](repeating: nil, count: count) }
                        func set(index: Int, shapes: [ShapeType]) {
                                lock.lock()
                                self.results[index] = shapes
                                lock.unlock()
                        }
                }

                let jobResults = JobResults(count: shapeBatches.count)
                let jobs = shapeBatches.map { $0.job }

                await withTaskGroup(of: Void.self) { group in
                        for index in 0..<jobs.count {
                                group.addTask {
                                        do {
                                                let shapes = try jobs[index]()
                                                jobResults.set(index: index, shapes: shapes)
                                        } catch {
                                                print("Error in concurrent shape building: \(error)")
                                        }
                                }
                        }
                }

                // Pre-allocate geometric arrays to prevent existential O(N) reallocation traps
                var totalShapesCount = 0
                for shapes in jobResults.results {
                        totalShapesCount += shapes?.count ?? 0
                }
                apiGeometricPrimitives.reserveCapacity(totalShapesCount)
                options.primitives.reserveCapacity(totalShapesCount)  // over-estimated but absolutely safe and prevents reallocation

                // Track large meshes that need their own local BVH
                var meshBvhPrims = [[GeometricPrimitive]]()

                for (index, batch) in shapeBatches.enumerated() {
                        guard let shapes = jobResults.results[index], !shapes.isEmpty else { continue }
                        var areaLightsBatch = [Light]()
                        var prims = [IntersectablePrimitive]()

                        if batch.isAreaLight {
                                for shape in shapes {
                                        guard
                                                batch.areaLightName == "area"
                                                        || batch.areaLightName == "diffuse"
                                        else { throw SceneDescriptionError.areaLight }
                                        guard
                                                let brightness = try batch.areaLightParameters.findSpectrum(
                                                        name: "L")
                                                        as? RgbSpectrum
                                        else {
                                                throw ParameterError.missing(
                                                        parameter: "L", function: #function)
                                        }
                                        let scale = try batch.areaLightParameters.findOneReal(
                                                called: "scale", else: 1)
                                        let scaledBrightness = brightness * scale
                                        let areaLight = AreaLight(
                                                brightness: scaledBrightness, shape: shape,
                                                alpha: batch.alpha,
                                                reverseOrientation: batch.reverseOrientation,
                                                idx: self.areaLights.count)
                                        let light = Light.area(areaLight)
                                        areaLightsBatch.append(light)
                                        prims.append(.areaLight(areaLight))
                                        self.areaLights.append(areaLight)
                                }

                                if let objectName = batch.objectName {
                                        if options.objects[objectName] == nil {
                                                options.objects[objectName] = prims
                                        } else {
                                                options.objects[objectName]!.append(contentsOf: prims)
                                        }
                                } else {
                                        options.primitives.append(contentsOf: prims)
                                        options.lights.append(contentsOf: areaLightsBatch)
                                }
                        } else {
                                var currentPrims = [GeometricPrimitive]()
                                currentPrims.reserveCapacity(shapes.count)
                                for shape in shapes {
                                        let geometricPrimitive = GeometricPrimitive(
                                                shape: shape, materialIndex: batch.materialIndex,
                                                mediumInterface: batch.currentMediumInterface,
                                                alpha: batch.alpha,
                                                reverseOrientation: batch.reverseOrientation,
                                                idx: apiGeometricPrimitives.count)
                                        currentPrims.append(geometricPrimitive)
                                        apiGeometricPrimitives.append(geometricPrimitive)
                                }

                                if let objectName = batch.objectName {
                                        let primitivesList: [IntersectablePrimitive] = currentPrims.map {
                                                .geometricPrimitive($0)
                                        }
                                        if options.objects[objectName] == nil {
                                                options.objects[objectName] = primitivesList
                                        } else {
                                                options.objects[objectName]!.append(
                                                        contentsOf: primitivesList)
                                        }
                                } else {
                                        if currentPrims.count > 16 {
                                                // It's a large top-level mesh. Buffer it for Pass 2 (Local BVH)
                                                meshBvhPrims.append(currentPrims)
                                        } else {
                                                options.primitives.append(
                                                        contentsOf: currentPrims.map {
                                                                .geometricPrimitive($0)
                                                        })
                                        }
                                }
                        }
                }
                shapeBatches.removeAll()

                // PASS 3: Build local Per-Mesh BVHs concurrently for massive performance
                if !meshBvhPrims.isEmpty {
                        final class LocalAccelerators: @unchecked Sendable {
                                var results: [Accelerator?]
                                let lock = NSLock()
                                init(count: Int) {
                                        self.results = [Accelerator?](repeating: nil, count: count)
                                }
                                func set(index: Int, accelerator: Accelerator) {
                                        lock.lock()
                                        self.results[index] = accelerator
                                        lock.unlock()
                                }
                        }

                        let localAccelerators = LocalAccelerators(count: meshBvhPrims.count)
                        let immutableMeshBvhPrims = meshBvhPrims
                        let localAcceleratorName = self.acceleratorName

                        let tempScene = Scene(
                                lights: [], materials: materials, meshes: triangleMeshBuilder.getMeshes(),
                                geometricPrimitives: apiGeometricPrimitives, areaLights: areaLights,
                                transformedPrimitives: [], arena: state.arena)

                        await withTaskGroup(of: Void.self) { group in
                                for index in 0..<immutableMeshBvhPrims.count {
                                        group.addTask {
                                                do {
                                                        let primsArray: [IntersectablePrimitive] =
                                                                immutableMeshBvhPrims[index].map {
                                                                        .geometricPrimitive($0)
                                                                }
                                                        let accelerator = try await makeAccelerator(
                                                                scene: tempScene, primitives: primsArray,
                                                                acceleratorName: localAcceleratorName)
                                                        localAccelerators.set(
                                                                index: index, accelerator: accelerator)
                                                } catch {
                                                        print("Error building local BVH: \(error)")
                                                }
                                        }
                                }
                        }

                        let identityTransform = Transform()
                        for accelerator in localAccelerators.results {
                                guard let acc = accelerator else { continue }
                                let tpFinal = TransformedPrimitive(
                                        accelerator: acc, transform: identityTransform,
                                        idx: transformedPrimitives.count)
                                transformedPrimitives.append(tpFinal)
                                options.primitives.append(.transformedPrimitive(tpFinal))
                        }
                }

                try await resolveInstances()

                readProgressTask?.cancel()
                _ = await readProgressTask?.value

                if renderOptions.justParse { return }
                let renderer = try await options.makeRenderer(
                        geometricPrimitives: apiGeometricPrimitives,
                        areaLights: areaLights,
                        materials: materials,
                        transformedPrimitives: transformedPrimitives,
                        acceleratorName: acceleratorName,
                        immutableState: state.getImmutable(),
                        renderOptions: renderOptions,
                        meshes: triangleMeshBuilder.getMeshes(),
                        arena: state.arena)
                try await renderer.render()
                self.options = RenderConfiguration()
        }

        private func makeLight(
                name: String,
                parameters: ParameterDictionary,
                lightToWorld: Transform
        )
                throws -> Light
        {
                switch name {
                case "distant":
                        let distantLight = try DistantLight.create(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                        return Light.distant(distantLight)
                case "infinite":
                        let infiniteLight = try InfiniteLight.create(
                                lightToWorld: lightToWorld,
                                parameters: parameters,
                                sceneDirectory: renderOptions.sceneDirectory, arena: &state.arena)
                        // immortalize(infiniteLight)
                        return Light.infinite(infiniteLight)
                case "point":
                        let pointLight = try PointLight.create(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                        return Light.point(pointLight)
                default:
                        throw SceneDescriptionError.makeLight(message: name)
                }
        }

}

func getTextureFrom(name: String, type: String, sceneDirectory: String, arena: inout TextureArena) throws
        -> Texture
{
        let fileManager = FileManager.default
        let absoluteFileName = sceneDirectory + "/" + name
        guard fileManager.fileExists(atPath: absoluteFileName) else {
                print("Warning: Can't find texture file: \(absoluteFileName)")
                let idx = arena.appendRgb(RgbSpectrumTexture.constantTexture(ConstantTexture(value: gray)))
                return Texture.rgbSpectrumTexture(idx)
        }
        let suffix = absoluteFileName.suffix(4)
        switch suffix {
        case ".ptx":
                let idx = arena.appendRgb(RgbSpectrumTexture.ptex(Ptex(path: absoluteFileName)))
                return Texture.rgbSpectrumTexture(idx)
        case ".exr", ".pfm", ".png", ".tga":
                switch type {
                case "spectrum", "color":
                        let idx = arena.appendRgb(
                                RgbSpectrumTexture.openImageIoTexture(
                                        try OpenImageIOTexture(path: absoluteFileName, type: type)))
                        return Texture.rgbSpectrumTexture(idx)
                case "float":
                        let idx = arena.appendFloat(
                                FloatTexture.openImageIoTexture(
                                        try OpenImageIOTexture(path: absoluteFileName, type: type)))
                        return Texture.floatTexture(idx)
                default:
                        throw RenderError.unimplemented(
                                function: #function, file: #filePath, line: #line, message: "")
                }
        default:
                throw SceneDescriptionError.unknownTextureFormat(suffix: String(suffix))
        }
}
