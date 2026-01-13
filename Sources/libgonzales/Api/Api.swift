@preconcurrency import Foundation

@MainActor
func makeAccelerator(scene: Scene, primitives: [any Boundable & Intersectable]) async throws -> Accelerator {
        switch acceleratorName {
        case "bvh":
                let builder = await BoundingHierarchyBuilder(scene: scene, primitives: primitives)
                let boundingHierarchy = try builder.getBoundingHierarchy()
                // let accelerator = Accelerator.boundingHierarchy(boundingHierarchy)
                let accelerator = Accelerator(boundingHierarchy: boundingHierarchy)
                return accelerator
        // case "embree":
        //        let builder = await EmbreeBuilder(primitives: primitives)
        //        let embree = builder.getAccelerator()
        //        let accelerator = Accelerator.embree(embree)
        //        return accelerator
        // case "optix":
        //        let optix = try Optix(primitives: primitives)
        //        let accelerator = Accelerator.optix(optix)
        //        return accelerator
        default:
                throw ApiError.accelerator
        }
}

@MainActor
func lookAtTransform(eye: Point, at: Point, up: Vector) throws -> Transform {
        let dir: Vector = normalized(at - eye)
        var up = normalized(up)
        let right = normalized(cross(up, dir))
        if length(right) == 0 { return Transform() }
        up = cross(dir, right)
        let matrix = Matrix(
                t00: right.x, t01: up.x, t02: dir.x, t03: eye.x,
                t10: right.y, t11: up.y, t12: dir.y, t13: eye.y,
                t20: right.z, t21: up.z, t22: dir.z, t23: eye.z,
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
}

extension Api {

        @MainActor
        func attributeBegin() throws {
                try transformBegin()
                states.append(state)
        }

        @MainActor
        func attributeEnd() throws {
                try transformEnd()
                guard let last = states.popLast() else {
                        throw ApiError.parseAttributeEnd
                }
                state = last
        }

        @MainActor
        func camera(name _: String, parameters: ParameterDictionary) throws {
                options.cameraName = "perspective"
                options.cameraParameters = parameters
                options.cameraToWorld = currentTransform.inverse
                namedCoordinateSystems["camera"] = options.cameraToWorld
        }

        @MainActor
        func coordinateSystem(name: String) {
                namedCoordinateSystems[name] = currentTransform
        }

        @MainActor
        func coordSysTransform(name: String) throws {
                guard let transform = namedCoordinateSystems[name] else {
                        throw ApiError.coordSysTransform
                }
                currentTransform = transform
        }

        @MainActor
        func concatTransform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform *= Transform(matrix: matrix)
        }

        @MainActor
        func film(name: String, parameters: ParameterDictionary) throws {
                let fileName = try parameters.findString(called: "filename") ?? "gonzales.exr"
                guard fileName.hasSuffix("exr") else {
                        abort("Only exr output supported!")
                }
                options.filmName = name
                options.filmParameters = parameters
        }

        @MainActor
        func identity() {
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
        func areaLight(name: String, parameters: ParameterDictionary) throws {
                guard name == "diffuse" || name == "area" else {
                        throw ApiError.areaLight
                }
                state.areaLight = name
                state.areaLightParameters = parameters
        }

        @MainActor
        func accelerator(name: String, parameters _: ParameterDictionary) throws {
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
        func integrator(name: String, parameters: ParameterDictionary) throws {
                options.integratorName = name
                options.integratorParameters = parameters
        }

        @MainActor
        func lightSource(name: String, parameters: ParameterDictionary) throws {
                let light = try makeLight(
                        name: name,
                        parameters: parameters,
                        lightToWorld: currentTransform)
                options.lights.append(light)
        }

        @MainActor
        func lookAt(eye: Point, at: Point, up: Vector) throws {
                let transform = try lookAtTransform(eye: eye, at: at, up: up)
                currentTransform *= transform
        }

        @MainActor
        func makeNamedMaterial(name: String, parameters: ParameterDictionary) throws {
                let type = try parameters.findString(called: "type") ?? "defaultMaterial"
                state.namedMaterials[name] = UninstancedMaterial(type: type, parameters: parameters)
        }

        @MainActor
        func makeNamedMedium(name: String, parameters: ParameterDictionary) throws {
                guard let type = try parameters.findString(called: "type") else {
                        throw ApiError.namedMedium
                }
                switch type {
                case "cloud":
                        warning("Cloud is not implemented!")
                case "homogeneous":
                        let scale = try parameters.findOneFloatX(called: "scale", else: 1)
                        let absorption =
                                try parameters.findSpectrum(name: "sigma_a", else: white) as! RgbSpectrum
                        let scattering =
                                try parameters.findSpectrum(name: "sigma_s", else: white) as! RgbSpectrum
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
        func material(type: String, parameters: ParameterDictionary) throws {
                state.currentMaterial = UninstancedMaterial(type: type, parameters: parameters)
        }

        @MainActor
        func mediumInterface(interior: String, exterior: String) {
                state.currentMediumInterface = MediumInterface(interior: interior, exterior: exterior)
        }

        @MainActor
        func namedMaterial(name: String) throws {
                state.currentNamedMaterial = name
        }

        @MainActor
        func objectBegin(name: String) throws {
                try attributeBegin()
                state.objectName = name
        }

        @MainActor
        func objectEnd() throws {
                try attributeEnd()
                state.objectName = nil
        }

        @MainActor
        func objectInstance(name _: String) async throws {
                unimplemented()
                // guard var primitives = options.objects[name] else {
                //        return
                // }
                // if primitives.isEmpty {
                //        return
                // }
                // let accelerator = try await makeAccelerator(scene: accessToSceneNeeded, primitives: primitives)
                // primitives.removeAll()
                // options.objects[name] = [accelerator]
                // let instance = TransformedPrimitive(
                //        accelerator: accelerator,
                //        transform: currentTransform,
                //        idx: transformedPrimitives.count)
                // options.primitives.append(instance)
                // transformedPrimitives.append(instance)
        }

        @MainActor
        func sampler(name: String, parameters: ParameterDictionary) {
                options.samplerName = name
                options.samplerParameters = parameters
        }

        @MainActor
        func pixelFilter(name: String, parameters: ParameterDictionary) {
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
        func transform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform = Transform(matrix: matrix)
        }

        @MainActor
        func transformBegin() throws {
                transforms.append(currentTransform)
        }

        @MainActor
        func transformEnd() throws {
                guard let last = transforms.popLast() else {
                        throw ApiError.transformsEmpty
                }
                currentTransform = last
        }

        @MainActor
        func scale(x: FloatX, y: FloatX, z: FloatX) throws {
                let matrix = Matrix(
                        t00: x, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: y, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: z, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                currentTransform *= Transform(matrix: matrix)
        }

        // Not really part of PBRT API
        @MainActor
        public func start() {
                readTimer = Timer("Reading...", newline: false)
                fflush(stdout)
        }

        @MainActor
        func rotate(by angle: FloatX, around axis: Vector) throws {
                let a = normalized(axis)
                let theta = radians(deg: angle)
                let sinTheta = sin(theta)
                let cosTheta = cos(theta)
                let t00 = a.x * a.x + (1 - a.x * a.x) * cosTheta
                let t01 = a.x * a.y * (1 - cosTheta) - a.z * sinTheta
                let t02 = a.x * a.z * (1 - cosTheta) + a.y * sinTheta
                let t10 = a.x * a.y * (1 - cosTheta) + a.z * sinTheta
                let t11 = a.y * a.y + (1 - a.y * a.y) * cosTheta
                let t12 = a.y * a.z * (1 - cosTheta) - a.x * sinTheta
                let t20 = a.x * a.z * (1 - cosTheta) - a.y * sinTheta
                let t21 = a.y * a.z * (1 - cosTheta) + a.x * sinTheta
                let t22 = a.z * a.z + (1 - a.z * a.z) * cosTheta
                let matrix = Matrix(
                        t00: t00, t01: t01, t02: t02, t03: 0,
                        t10: t10, t11: t11, t12: t12, t13: 0,
                        t20: t20, t21: t21, t22: t22, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                currentTransform *= Transform(matrix: matrix)
        }

        @MainActor
        func texture(
                name: String,
                type: String,
                textureClass: String,
                parameters: ParameterDictionary
        )
                throws
        {
                guard type == "spectrum" || type == "float" || type == "color" else {
                        warning("Unimplemented texture type: \(type)")
                        return
                }
                var texture: Texture
                switch textureClass {
                // bilerp missing
                case "checkerboard":
                        unimplemented()
                // let rgbSpectrumTextureEven = try parameters.findRgbSpectrumTexture(name: "tex1")
                // let textureEven = Texture.rgbSpectrumTexture(rgbSpectrumTextureEven)
                // let rgbSpectrumTextureOdd = try parameters.findRgbSpectrumTexture(name: "tex2")
                // let textureOdd = Texture.rgbSpectrumTexture(rgbSpectrumTextureOdd)
                // let textures = (textureEven, textureOdd)
                // let uscale = try parameters.findOneFloatX(called: "uscale", else: 1)
                // let vscale = try parameters.findOneFloatX(called: "vscale", else: 1)
                // let scale = (uscale, vscale)
                // let checkerboard = Checkerboard(textures: textures, scale: scale)
                // let rgbSpectrumTexture = RgbSpectrumTexture.checkerboard(checkerboard)
                // texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
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
                // directionmix missing
                // dots missing
                // fbm missing
                case "imagemap":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                // marble missing
                case "mix":
                        switch type {
                        case "color", "spectrum":
                                unimplemented()
                        // let tex1 = try parameters.findRgbSpectrumTexture(name: "tex1")
                        // let tex2 = try parameters.findRgbSpectrumTexture(name: "tex2")
                        // let amount = try parameters.findOneFloatX(called: "amount", else: 0.5)
                        // let rgbSpectrumMixTexture = RgbSpectrumMixTexture(
                        //        textures: (tex1, tex2), amount: amount)
                        // let rgbSpectrumTexture = RgbSpectrumTexture.rgbSpectrumMixTexture(
                        //        rgbSpectrumMixTexture)
                        // texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                        case "float":
                                // let tex1 = try parameters.findFloatXTexture(name: "tex1")
                                // let tex2 = try parameters.findFloatXTexture(name: "tex2")
                                // let amount = try parameters.findOneFloatX(called: "amount", else: 0.5)
                                // let floatMixTexture = FloatMixTexture(textures: (tex1, tex2), amount: amount)
                                // let floatTexture = FloatTexture.floatMixTexture(floatMixTexture)
                                // texture = Texture.floatTexture(floatTexture)
                                unimplemented()
                        default:
                                unimplemented()
                        }
                case "ptex":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                case "scale":
                        unimplemented()
                // let scaleFloatTexture = try parameters.findFloatXTexture(name: "scale")
                // let scale = Texture.floatTexture(scaleFloatTexture)
                // let texRgbSpectrumTexture = try parameters.findRgbSpectrumTexture(name: "tex")
                // let tex = Texture.rgbSpectrumTexture(texRgbSpectrumTexture)
                // let scaledTexture = ScaledTexture(scale: scale, texture: tex)
                // let rgbSpectrumTexture = RgbSpectrumTexture.scaledTexture(scaledTexture)
                // texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                // windy missing
                // wrinkled missing
                default:
                        warning("Unimplemented texture class: \(textureClass)")
                        return
                }
                state.textures[name] = texture
        }

        @MainActor
        func translate(by: Vector) throws {
                let matrix = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: by.x,
                        t10: 0, t11: 1, t12: 0, t13: by.y,
                        t20: 0, t21: 0, t22: 1, t23: by.z,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let translation = Transform(matrix: matrix)
                currentTransform *= translation
        }

        @MainActor
        public func worldBegin() {
                currentTransform = Transform()
        }

        func dumpCamera(_ camera: any Camera) {
                if let perspectiveCamera = camera as? PerspectiveCamera {
                        print(perspectiveCamera.objectToWorld)
                }
        }

        @MainActor
        func worldEnd() async throws {
                print("Reading: \(readTimer.elapsed)")
                if renderOptions.justParse { return }
                let renderer = try await options.makeRenderer(
                        geometricPrimitives: apiGeometricPrimitives, areaLights: areaLights)
                try await renderer.render()
                cleanUp()
        }

        @MainActor
        private func cleanUp() {
                options = Options()
        }

        @MainActor
        private func makeLight(
                name: String,
                parameters: ParameterDictionary,
                lightToWorld: Transform
        )
                throws -> Light
        {
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
                throws -> [ShapeType]
        {
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
var options = Options()

@MainActor
var state = State()

@MainActor
var states = [State]()

@MainActor
var currentTransform = Transform()

@MainActor
var transforms = [Transform]()

@MainActor
var readTimer = Timer("")

@MainActor
var acceleratorName = "bvh"

@MainActor
var namedCoordinateSystems = [String: Transform]()
