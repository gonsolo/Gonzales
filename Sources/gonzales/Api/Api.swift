import Foundation

func makeAccelerator(primitives: [Boundable & Intersectable]) throws -> Accelerator {
        switch acceleratorName {
        case "bvh":
                let builder = BoundingHierarchyBuilder(primitives: primitives)
                return try builder.getBoundingHierarchy()
        case "embree":
                let builder = EmbreeBuilder(primitives: primitives)
                return builder.getAccelerator()
        default:
                throw ApiError.accelerator
        }
}

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

enum ApiError: Error {
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

extension TimeInterval {
        public var humanReadable: String {
                let seconds = self.truncatingRemainder(dividingBy: 60)
                if self > 60 {
                        let minutes = self / 60
                        let m = String(format: "%.0f", minutes)
                        let s = String(format: "%.0f", seconds)
                        return "\(m)m\(s)s"
                } else {
                        let s = String(format: "%.1f", seconds)
                        return "\(s)s"
                }
        }
}

struct Api {

        func attributeBegin() throws {
                try transformBegin()
                states.append(state)
        }

        func attributeEnd() throws {
                try transformEnd()
                guard let last = states.popLast() else {
                        throw ApiError.parseAttributeEnd
                }
                state = last
        }

        func camera(name: String, parameters: ParameterDictionary) throws {
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
                        throw ApiError.coordSysTransform
                }
                currentTransform = transform
        }

        func concatTransform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform *= Transform(matrix: matrix)
        }

        func film(name: String, parameters: ParameterDictionary) throws {
                let fileName = try parameters.findString(called: "filename") ?? "gonzales.exr"
                guard fileName.hasSuffix("exr") else {
                        abort("Only exr output supported!")
                }
                options.filmName = name
                options.filmParameters = parameters
        }

        func identity() {
                currentTransform = Transform()
        }

        mutating func importFile(file sceneName: String) throws {
                try include(file: sceneName, render: false)
        }

        mutating func include(file sceneName: String, render: Bool) throws {
                print(sceneName)
                do {
                        let fileManager = FileManager.default
                        let absoluteSceneName = sceneDirectory + "/" + sceneName
                        var components = absoluteSceneName.components(separatedBy: ".")
                        components.removeLast()
                        guard fileManager.fileExists(atPath: absoluteSceneName) else {
                                throw RenderError.fileNotExisting(name: absoluteSceneName)
                        }
                        if #available(OSX 10.15, *) {
                                let parser = try Parser(fileName: absoluteSceneName, render: render)
                                try parser.parse()
                        } else {
                                // Fallback on earlier versions
                        }
                } catch ApiError.wrongType(let message) {
                        print("Error: Wrong type: \(message) in \(sceneName).")
                }
        }

        func areaLight(name: String, parameters: ParameterDictionary) throws {
                guard name == "diffuse" || name == "area" else {
                        throw ApiError.areaLight
                }
                state.areaLight = name
                state.areaLightParameters = parameters
        }

        func accelerator(name: String, parameters: ParameterDictionary) throws {
                switch name {
                case "bvh":
                        acceleratorName = "bvh"
                case "embree":
                        acceleratorName = "embree"
                default:
                        throw ApiError.accelerator
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

        func lookAt(eye: Point, at: Point, up: Vector) throws {
                let transform = try lookAtTransform(eye: eye, at: at, up: up)
                currentTransform *= transform
        }

        func makeNamedMaterial(name: String, parameters: ParameterDictionary) throws {
                let type = try parameters.findString(called: "type") ?? "defaultMaterial"
                state.namedMaterials[name] = UninstancedMaterial(type: type, parameters: parameters)
        }

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
                                try parameters.findSpectrum(name: "sigma_a", else: white) as! RGBSpectrum
                        let scattering =
                                try parameters.findSpectrum(name: "sigma_s", else: white) as! RGBSpectrum
                        state.namedMedia[name] = Homogeneous(
                                scale: scale,
                                absorption: absorption,
                                scattering: scattering)
                case "nanovdb":
                        warning("Nanovdb is not implemented!")
                default:
                        throw ApiError.namedMedium
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
                guard var primitives = options.objects[name] else {
                        return
                }
                if primitives.isEmpty {
                        return
                }
                var instance: Boundable & Intersectable
                if primitives.count > 1 {
                        let accelerator = try makeAccelerator(primitives: primitives)
                        primitives.removeAll()
                        options.objects[name] = [accelerator]
                        instance = TransformedPrimitive(
                                primitive: accelerator,
                                transform: currentTransform)
                } else {
                        guard let first = primitives.first else {
                                warning("No primitive in objectInstance!")
                                return
                        }
                        instance = TransformedPrimitive(
                                primitive: first,
                                transform: currentTransform)
                }
                options.primitives.append(instance)
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
                var areaLights = [AreaLight]()
                var prims = [Boundable & Intersectable]()
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
                                                        name: "L") as? RGBSpectrum
                                else {
                                        throw ParameterError.missing(parameter: "L", function: #function)
                                }
                                let areaLight = AreaLight(brightness: brightness, shape: shape, alpha: alpha)
                                areaLights.append(areaLight)
                                prims.append(areaLight)
                        }
                } else {
                        materials[materialCounter] = material
                        for shape in shapes {
                                let geometricPrimitive = GeometricPrimitive(
                                        shape: shape,
                                        material: materialCounter,
                                        mediumInterface: state.currentMediumInterface,
                                        alpha: alpha)
                                prims.append(geometricPrimitive)
                        }
                        materialCounter = materialCounter + 1
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

        func transform(values: [FloatX]) throws {
                let matrix = Matrix(
                        t00: values[0], t01: values[4], t02: values[8], t03: values[12],
                        t10: values[1], t11: values[5], t12: values[9], t13: values[13],
                        t20: values[2], t21: values[6], t22: values[10], t23: values[14],
                        t30: values[3], t31: values[7], t32: values[11], t33: values[15])
                currentTransform = Transform(matrix: matrix)
        }

        func transformBegin() throws {
                transforms.append(currentTransform)
        }

        func transformEnd() throws {
                guard let last = transforms.popLast() else {
                        throw ApiError.transformsEmpty
                }
                currentTransform = last
        }

        func scale(x: FloatX, y: FloatX, z: FloatX) throws {
                let matrix = Matrix(
                        t00: x, t01: 0, t02: 0, t03: 0,
                        t10: 0, t11: y, t12: 0, t13: 0,
                        t20: 0, t21: 0, t22: z, t23: 0,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                currentTransform *= Transform(matrix: matrix)
        }

        // Not really part of PBRT API
        func start() {
                readTimer = Timer("Reading...", newline: false)
                fflush(stdout)
        }

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
                case "checkerboard":
                        let textureEven = try parameters.findRGBSpectrumTexture(name: "tex1")
                        let textureOdd = try parameters.findRGBSpectrumTexture(name: "tex2")
                        let textures = (textureEven, textureOdd)
                        let uscale = try parameters.findOneFloatX(called: "uscale", else: 1)
                        let vscale = try parameters.findOneFloatX(called: "vscale", else: 1)
                        let scale = (uscale, vscale)
                        texture = Checkerboard(textures: textures, scale: scale)
                case "constant":
                        texture = try parameters.findRGBSpectrumTexture(name: "value")
                case "imagemap":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                case "ptex":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName, type: type)
                case "scale":
                        //print(name)
                        //print(parameters)
                        //let scale = try parameters.findOneFloatX(called: "scale", else: 1)
                        //print("scale: ", scale)
                        //let scaleTexture = try parameters.findTexture(name: "scale")
                        //print("scaleTexture: ", scaleTexture)
                        let scale = try parameters.findFloatXTexture(name: "scale")
                        //print("scale: ", scale)

                        let tex = try parameters.findRGBSpectrumTexture(name: "tex")
                        //print("tex: ", tex)

                        //guard let unscaledTexture = state.textures[tex] else {
                        //        throw ApiError.unknownTexture(name: tex)
                        //}
                        texture = ScaledTexture(scale: scale, texture: tex)
                default:
                        warning("Unimplemented texture class: \(textureClass)")
                        return
                }
                state.textures[name] = texture
        }

        func translate(by: Vector) throws {
                let matrix = Matrix(
                        t00: 1, t01: 0, t02: 0, t03: by.x,
                        t10: 0, t11: 1, t12: 0, t13: by.y,
                        t20: 0, t21: 0, t22: 1, t23: by.z,
                        t30: 0, t31: 0, t32: 0, t33: 1)
                let translation = Transform(matrix: matrix)
                currentTransform *= translation
        }

        public func worldBegin() {
                currentTransform = Transform()
        }

        func dumpPoints(_ type: String, _ triangle: Triangle) {
                let points = triangle.getLocalPoints()
                print(
                        type, " { ",
                        "{", points.0.x, ", ", points.0.y, ", ", points.0.z, "},",
                        "{", points.1.x, ", ", points.1.y, ", ", points.1.z, "},",
                        "{", points.2.x, ", ", points.2.y, ", ", points.2.z, "}",
                        " },")
        }

        func dumpPrimitives() {
                print("\nNumber of primitives: ", options.primitives.count)
                for primitive in options.primitives {
                        var type = ""
                        var shape: Shape? = nil
                        if let geom = primitive as? GeometricPrimitive {
                                type = "GeometricPrimitive"
                                shape = geom.shape
                        } else if let areaLight = primitive as? AreaLight {
                                type = "AreaLight"
                                shape = areaLight.shape
                        } else {
                                print("not geom: ", primitive)
                        }
                        if let triangle = shape as? Triangle {
                                dumpPoints(type, triangle)
                        }
                }
        }

        func dumpCamera(_ camera: Camera) {
                if let perspectiveCamera = camera as? PerspectiveCamera {
                        print(perspectiveCamera.objectToWorld)
                }
        }

        func worldEnd() throws {
                print("Reading: \(readTimer.elapsed)")
                if justParse { return }
                let renderer = try options.makeRenderer()
                try renderer.render()
                if verbose { statistics.report() }
                cleanUp()
        }

        private func cleanUp() {
                options = Options()
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
                        return try createDistantLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                case "infinite":
                        return try createInfiniteLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                case "point":
                        return try createPointLight(
                                lightToWorld: lightToWorld,
                                parameters: parameters)
                default:
                        throw ApiError.makeLight(message: name)
                }
        }

        private func makeDefaultMaterial(insteadOf material: String) throws -> Material {
                warnOnce("Unknown material \"\(material)\". Creating default.")
                var parameters = ParameterDictionary()
                parameters["reflectance"] = [gray]
                return try createDiffuse(parameters: parameters)
        }

        func makeMaterial(type: String, parameters: ParameterDictionary) throws -> Material {

                var material: Material
                switch type {
                case "coateddiffuse":
                        material = try createCoatedDiffuse(parameters: parameters)
                // coatedconductor missing
                case "conductor":
                        material = try createConductor(parameters: parameters)
                case "dielectric":
                        material = try createDielectric(parameters: parameters)
                case "diffuse":
                        material = try createDiffuse(parameters: parameters)
                case "diffusetransmission":
                        material = try createDiffuseTransmission(parameters: parameters)
                case "hair":
                        material = try createHair(parameters: parameters)
                case "interface":
                        material = try createInterface(parameters: parameters)
                case "measured":
                        material = try createMeasured(parameters: parameters)
                // mix missing
                // subsurface missing
                // thindielectric missing
                default:
                        material = try makeDefaultMaterial(insteadOf: type)
                }
                return material
        }

        private func makeShapes(
                name: String,
                objectToWorld: Transform,
                parameters: ParameterDictionary
        )
                throws -> [Shape]
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
                                parameters: parameters)  // TODO
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

func getTextureFrom(name: String, type: String) throws -> Texture {
        let fileManager = FileManager.default
        let absoluteFileName = sceneDirectory + "/" + name
        guard fileManager.fileExists(atPath: absoluteFileName) else {
                warning("Can't find texture file: \(absoluteFileName)")
                return ConstantTexture(value: gray)
        }
        let suffix = absoluteFileName.suffix(4)
        switch suffix {
        case ".ptx":
                return Ptex(path: absoluteFileName)
        case ".exr", ".pfm", ".png", ".tga":
                return OpenImageIOTexture(path: absoluteFileName, type: type)
        default:
                throw ApiError.unknownTextureFormat(suffix: String(suffix))
        }
}

var api = Api()
var options = Options()
var state = State()
var states = [State]()
var currentTransform = Transform()
var transforms = [Transform]()
var readTimer = Timer("")
var acceleratorName = "embree"
var namedCoordinateSystems = [String: Transform]()
