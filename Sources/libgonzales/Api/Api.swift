@preconcurrency import Foundation

@MainActor
func makeAccelerator(scene: Scene, primitives: [any Boundable & Intersectable], acceleratorName: String) async throws -> Accelerator {
        switch acceleratorName {
        case "bvh":
                let builder = await BoundingHierarchyBuilder(scene: scene, primitives: primitives)
                let boundingHierarchy = try builder.getBoundingHierarchy()
                let accelerator = Accelerator(boundingHierarchy: boundingHierarchy)
                return accelerator
        default:
                throw ApiError.accelerator
        }
}

@MainActor
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
        let transform = Transform(matrix: matrix.inverse)
        return transform
}

public enum ApiError: Error {
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

public struct Api {
        var apiGeometricPrimitives = [GeometricPrimitive]()
        var areaLights = [AreaLight]()
        var acceleratorName = "bvh"
        var currentTransform = Transform()
        var namedCoordinateSystems = [String: Transform]()
        var options: Options
        var readTimer: Timer?
        var state: State
        var states = [State]()
        var transforms = [Transform]()

        @MainActor
        public init() {
                self.options = Options()
                self.state = State()
        }
}

extension Api {

        @MainActor
        mutating func attributeBegin() throws {
                try transformBegin()
                states.append(state)
        }

        @MainActor
        mutating func attributeEnd() throws {
                try transformEnd()
                guard let last = states.popLast() else {
                        throw ApiError.parseAttributeEnd
                }
                state = last
        }

        @MainActor
        mutating func camera(name _: String, parameters: ParameterDictionary) throws {
                options.cameraName = "perspective"
                options.cameraParameters = parameters
                options.cameraToWorld = currentTransform.inverse
                namedCoordinateSystems["camera"] = options.cameraToWorld
        }

        @MainActor
        mutating func coordinateSystem(name: String) {
                namedCoordinateSystems[name] = currentTransform
        }

        @MainActor
        mutating func coordSysTransform(name: String) throws {
                guard let transform = namedCoordinateSystems[name] else {
                        throw ApiError.coordSysTransform
                }
                currentTransform = transform
        }

        @MainActor
        mutating func concatTransform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform *= Transform(matrix: matrix)
        }

        @MainActor
        mutating func film(name: String, parameters: ParameterDictionary) throws {
                let fileName = try parameters.findString(called: "filename") ?? "gonzales.exr"
                guard fileName.hasSuffix("exr") else {
                        abort("Only exr output supported!")
                }
                options.filmName = name
                options.filmParameters = parameters
        }

        @MainActor
        mutating func identity() {
                currentTransform = Transform()
        }

        @MainActor
        func importFile(file sceneName: String) async throws {
                try await include(file: sceneName, render: false)
        }

        @MainActor
        public func include(file sceneName: String, render: Bool) async throws {
                print(sceneName)
                do {
                        let fileManager = FileManager.default
                        let absoluteSceneName = renderOptions.sceneDirectory + "/" + sceneName
                        var components = absoluteSceneName.components(separatedBy: ".")
                        components.removeLast()
                        guard fileManager.fileExists(atPath: absoluteSceneName) else {
                                throw RenderError.fileNotExisting(name: absoluteSceneName)
                        }
                        if #available(OSX 10.15, *) {
                                let parser = try Parser(fileName: absoluteSceneName, render: render)
                                try await parser.parse()
                        } else {
                                // Fallback on earlier versions
                        }
                } catch ApiError.wrongType(let message) {
                        print("Error: Wrong type: \(message) in \(sceneName).")
                }
        }

        @MainActor
        mutating func areaLight(name: String, parameters: ParameterDictionary) throws {
                guard name == "diffuse" || name == "area" else {
                        throw ApiError.areaLight
                }
                state.areaLight = name
                state.areaLightParameters = parameters
        }

        @MainActor
        mutating func accelerator(name: String, parameters _: ParameterDictionary) throws {
                switch name {
                case "bvh":
                        acceleratorName = "bvh"
                case "embree":
                        acceleratorName = "embree"
                case "optix":
                        acceleratorName = "optix"
                default:
                        throw ApiError.accelerator
                }
        }

        @MainActor
        mutating func integrator(name: String, parameters: ParameterDictionary) throws {
                options.integratorName = name
                options.integratorParameters = parameters
        }

        @MainActor
        mutating func lightSource(name: String, parameters: ParameterDictionary) throws {
                let light = try makeLight(
                        name: name,
                        parameters: parameters,
                        lightToWorld: currentTransform)
                options.lights.append(light)
        }

        @MainActor
        mutating func lookAt(eye: Point, target: Point, upVector: Vector) throws {
                let transform = try lookAtTransform(eye: eye, target: target, upVector: upVector)
                currentTransform *= transform
        }

        @MainActor
        mutating func makeNamedMaterial(name: String, parameters: ParameterDictionary) throws {
                let type = try parameters.findString(called: "type") ?? "defaultMaterial"
                state.namedMaterials[name] = UninstancedMaterial(type: type, parameters: parameters)
        }

        @MainActor
        mutating func makeNamedMedium(name: String, parameters: ParameterDictionary) throws {
                guard let type = try parameters.findString(called: "type") else {
                        throw ApiError.namedMedium
                }
                switch type {
                case "cloud":
                        warning("Cloud is not implemented!")
                case "homogeneous":
                        let scale = try parameters.findOneFloatX(called: "scale", else: 1)
                        let absorption =
                                try parameters.findSpectrum(name: "sigma_a", else: white)?.asRgb() ?? white
                        let scattering =
                                try parameters.findSpectrum(name: "sigma_s", else: white)?.asRgb() ?? white
                        state.namedMedia[name] = Homogeneous(
                                scale: scale,
                                absorption: absorption,
                                scattering: scattering)
                case "nanovdb":
                        warning("Nanovdb is not implemented!")
                case "uniformgrid":
                        warning("Uniform grid is not implemented!")
                default:
                        throw ApiError.namedMedium
                }
        }

        @MainActor
        mutating func material(type: String, parameters: ParameterDictionary) throws {
                state.currentMaterial = UninstancedMaterial(type: type, parameters: parameters)
        }

        @MainActor
        mutating func mediumInterface(interior: String, exterior: String) {
                state.currentMediumInterface = MediumInterface(interior: interior, exterior: exterior)
        }

        @MainActor
        mutating func namedMaterial(name: String) throws {
                state.currentNamedMaterial = name
        }

        @MainActor
        mutating func objectBegin(name: String) throws {
                try attributeBegin()
                state.objectName = name
        }

        @MainActor
        mutating func objectEnd() throws {
                try attributeEnd()
                state.objectName = nil
        }

        @MainActor
        func objectInstance(name _: String) async throws {
                unimplemented()
        }

        @MainActor
        mutating func sampler(name: String, parameters: ParameterDictionary) {
                options.samplerName = name
                options.samplerParameters = parameters
        }

        @MainActor
        mutating func pixelFilter(name: String, parameters: ParameterDictionary) {
                options.filterName = name
                options.filterParameters = parameters
        }

        @MainActor
        mutating func shape(name: String, parameters: ParameterDictionary) throws {
                var areaLights = [Light]()
                var prims = [any Boundable & Intersectable]()
                let shapes = try makeShapes(
                        name: name,
                        objectToWorld: currentTransform,
                        parameters: parameters)
                if shapes.isEmpty {
                        return
                }
                let material = try state.createMaterial(parameters: parameters)
                let alpha = try parameters.findOneFloatX(called: "alpha", else: 1)
                if !state.areaLight.isEmpty {
                        for shape in shapes {
                                guard state.areaLight == "area" || state.areaLight == "diffuse"
                                else {
                                        throw ApiError.areaLight
                                }
                                guard
                                        let brightness =
                                                try state.areaLightParameters.findSpectrum(
                                                        name: "L") as? RgbSpectrum
                                else {
                                        throw ParameterError.missing(parameter: "L", function: #function)
                                }
                                let areaLight = AreaLight(
                                        brightness: brightness,
                                        shape: shape,
                                        alpha: alpha,
                                        idx: areaLights.count)
                                let light = Light.area(areaLight)
                                areaLights.append(light)
                                prims.append(areaLight)
                                self.areaLights.append(areaLight)
                        }
                } else {
                        let materialIndex = materials.count
                        materials.append(material)
                        for shape in shapes {
                                let geometricPrimitive = GeometricPrimitive(
                                        shape: shape,
                                        materialIndex: materialIndex,
                                        mediumInterface: state.currentMediumInterface,
                                        alpha: alpha,
                                        idx: apiGeometricPrimitives.count)
                                prims.append(geometricPrimitive)
                                apiGeometricPrimitives.append(geometricPrimitive)
                        }
                }
                if let objectName = state.objectName {
                        if options.objects[objectName] == nil {
                                options.objects[objectName] = prims
                        } else {
                                options.objects[objectName]!.append(contentsOf: prims)
                        }
                } else {
                        options.primitives.append(contentsOf: prims)
                        options.lights.append(contentsOf: areaLights)
                }
        }

        @MainActor
        mutating func transform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform = Transform(matrix: matrix)
        }

        @MainActor
        mutating func transformBegin() throws {
                transforms.append(currentTransform)
        }

        @MainActor
        mutating func transformEnd() throws {
                guard let last = transforms.popLast() else {
                        throw ApiError.transformsEmpty
                }
                currentTransform = last
        }

        @MainActor
        mutating func scale(x: FloatX, y: FloatX, z: FloatX) throws {
                let matrix = Matrix(
                        t00: x, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: y, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: z, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                currentTransform *= Transform(matrix: matrix)
        }

        // Not really part of PBRT API
        @MainActor
        public mutating func start() {
                readTimer = Timer("Reading...", newline: false)
                fflush(stdout)
        }

        @MainActor
        mutating func rotate(by angle: FloatX, around axis: Vector) throws {
                let normalizedAxis = normalized(axis)
                let theta = radians(deg: angle)
                let sinTheta = sin(theta)
                let cosTheta = cos(theta)
                let t00 = normalizedAxis.x * normalizedAxis.x + (1 - normalizedAxis.x * normalizedAxis.x) * cosTheta
                let t01 = normalizedAxis.x * normalizedAxis.y * (1 - cosTheta) - normalizedAxis.z * sinTheta
                let t02 = normalizedAxis.x * normalizedAxis.z * (1 - cosTheta) + normalizedAxis.y * sinTheta
                let t10 = normalizedAxis.x * normalizedAxis.y * (1 - cosTheta) + normalizedAxis.z * sinTheta
                let t11 = normalizedAxis.y * normalizedAxis.y + (1 - normalizedAxis.y * normalizedAxis.y) * cosTheta
                let t12 = normalizedAxis.y * normalizedAxis.z * (1 - cosTheta) - normalizedAxis.x * sinTheta
                let t20 = normalizedAxis.x * normalizedAxis.z * (1 - cosTheta) - normalizedAxis.y * sinTheta
                let t21 = normalizedAxis.y * normalizedAxis.z * (1 - cosTheta) + normalizedAxis.x * sinTheta
                let t22 = normalizedAxis.z * normalizedAxis.z + (1 - normalizedAxis.z * normalizedAxis.z) * cosTheta
                let matrix = Matrix(
                        t00: t00, t01: t01, t02: t02, t03: 0,
                        t10: t10, t11: t11, t12: t12, t13: 0,
                        t20: t20, t21: t21, t22: t22, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                currentTransform *= Transform(matrix: matrix)
        }

        @MainActor
        mutating func texture(
                name: String,
                type: String,
                textureClass: String,
                parameters: ParameterDictionary
        )
                throws {
                guard type == "spectrum" || type == "float" || type == "color" else {
                        warning("Unimplemented texture type: \(type)")
                        return
                }
                var texture: Texture
                switch textureClass {
                case "checkerboard":
                        unimplemented()
                case "constant":
                        switch type {
                        case "spectrum", "color":
                                let rgbSpectrumTexture = try parameters.findRgbSpectrumTexture(name: "value")
                                texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                        case "float":
                                let floatTexture = try parameters.findFloatXTexture(name: "value")
                                texture = Texture.floatTexture(floatTexture)
                        default:
                                unimplemented()
                        }
                case "imagemap":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                case "mix":
                        switch type {
                        case "color", "spectrum":
                                unimplemented()
                        case "float":
                                unimplemented()
                        default:
                                unimplemented()
                        }
                case "ptex":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                case "scale":
                        unimplemented()
                default:
                        warning("Unimplemented texture class: \(textureClass)")
                        return
                }
                state.textures[name] = texture
        }

        @MainActor
        mutating func translate(amount: Vector) throws {
                let matrix = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: amount.x,
                        t10: 0, t11: 1, t12: 0, t13: amount.y,
                        t20: 0, t21: 0, t22: 1, t23: amount.z,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let translation = Transform(matrix: matrix)
                currentTransform *= translation
        }

        @MainActor
        public mutating func worldBegin() {
                currentTransform = Transform()
        }



        @MainActor
        func worldEnd() async throws {
                print("Reading: \(readTimer?.elapsed ?? "unknown")")
                if renderOptions.justParse { return }
                let renderer = try await options.makeRenderer(
                        geometricPrimitives: apiGeometricPrimitives, areaLights: areaLights)
                try await renderer.render()
                api.options = Options()
        }

        @MainActor
        private func makeLight(
                name: String,
                parameters: ParameterDictionary,
                lightToWorld: Transform
        )
                throws -> Light {
                switch name {
                case "distant":
                        let distantLight = try createDistantLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                        return Light.distant(distantLight)
                case "infinite":
                        let infiniteLight = try createInfiniteLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                        // immortalize(infiniteLight)
                        return Light.infinite(infiniteLight)
                case "point":
                        let pointLight = try createPointLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                        return Light.point(pointLight)
                default:
                        throw ApiError.makeLight(message: name)
                }
        }

        @MainActor
        private func makeShapes(
                name: String,
                objectToWorld: Transform,
                parameters: ParameterDictionary
        )
                throws -> [ShapeType] {
                switch name {
                case "bilinearmesh":
                        return []  // Ignore for now
                case "curve":
                        return try createCurveShape(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
                case "cylinder":
                        return []  // Ignore for now
                case "disk":
                        return try createDiskShape(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
                case "loopsubdiv":
                        return try createTriangleMeshShape(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
                case "plymesh":
                        return try createPlyMesh(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
                case "sphere":
                        return [
                                try createSphere(
                                        objectToWorld: objectToWorld,
                                        parameters: parameters)
                        ]
                case "trianglemesh":
                        return try createTriangleMeshShape(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
                default:
                        throw ApiError.makeShapes(message: name)
                }
        }
}

@MainActor
func getTextureFrom(name: String, type: String) throws -> Texture {
        let fileManager = FileManager.default
        let absoluteFileName = renderOptions.sceneDirectory + "/" + name
        guard fileManager.fileExists(atPath: absoluteFileName) else {
                warning("Can't find texture file: \(absoluteFileName)")
                return Texture.rgbSpectrumTexture(
                        RgbSpectrumTexture.constantTexture(ConstantTexture(value: gray)))
        }
        let suffix = absoluteFileName.suffix(4)
        switch suffix {
        case ".ptx":
                return Texture.rgbSpectrumTexture(RgbSpectrumTexture.ptex(Ptex(path: absoluteFileName)))
        case ".exr", ".pfm", ".png", ".tga":
                switch type {
                case "spectrum", "color":
                        return Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.openImageIoTexture(
                                        OpenImageIOTexture(path: absoluteFileName, type: type)))
                case "float":
                        return Texture.floatTexture(
                                FloatTexture.openImageIoTexture(
                                        OpenImageIOTexture(path: absoluteFileName, type: type)))
                default:
                        unimplemented()
                }
        default:
                throw ApiError.unknownTextureFormat(suffix: String(suffix))
        }
}

@MainActor
public var api = Api()

@MainActor
var materials = [Material]()
