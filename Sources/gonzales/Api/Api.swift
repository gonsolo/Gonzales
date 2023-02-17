import Foundation

func makeAccelerator(primitives: inout [Boundable & Intersectable]) throws -> Accelerator {
        switch acceleratorName {
        case "bvh":
                let builder = BoundingHierarchyBuilder(primitives: primitives)
                return try builder.getBoundingHierarchy()
        case "embree":
                return Embree(primitives: &primitives)
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
        case input(message: String)
        case makeLight(message: String)
        case makeSampler
        case makeShapes(message: String)
        case namedMaterial
        case objectInstance
        case parseAttributeEnd
        case ply(message: String)
        case unknownTextureFormat(suffix: String)
        case transformsEmpty
        case unknownAccelerator
        case unknownIntegrator
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

func getTextureFrom(name: String) throws -> RGBSpectrumTexture {
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
        case ".exr", ".tga", ".pfm":
                return OpenImageIOTexture(path: absoluteFileName)
        default:
                throw ApiError.unknownTextureFormat(suffix: String(suffix))
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
                guard let fileName = try parameters.findString(called: "filename") else {
                        abort("Missing filename")
                }
                guard fileName.hasSuffix("exr") else {
                        abort("Only exr output supported!")
                }
                options.filmName = name
                options.filmParameters = parameters
        }

        mutating func importFile(file sceneName: String) throws {
                try include(file: sceneName, render: false)
        }

        mutating func include(file sceneName: String, render: Bool) throws {
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
                // TODO
        }

        func material(type: String, parameters: ParameterDictionary) throws {
                state.currentMaterial = UninstancedMaterial(type: type, parameters: parameters)
        }

        func mediumInterface(interior: String, exterior: String) {
                // TODO
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
                        let accelerator = try makeAccelerator(primitives: &primitives)
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
                primitives.append(instance)
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
                                let areaLight = AreaLight(brightness: brightness, shape: shape)
                                areaLights.append(areaLight)
                                prims.append(areaLight)
                        }
                } else {
                        if !(material is None) {
                                materials[materialCounter] = material
                                for shape in shapes {
                                        let geometricPrimitive = GeometricPrimitive(
                                                shape: shape,
                                                material: materialCounter)
                                        prims.append(geometricPrimitive)
                                }
                                materialCounter = materialCounter + 1
                        }
                }
                if let name = state.objectName {
                        if options.objects[name] == nil {
                                if !(material is None) {
                                        options.objects[name] = prims
                                }
                        } else {
                                if !(material is None) {
                                        options.objects[name]!.append(contentsOf: prims)
                                }
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
                var texture: RGBSpectrumTexture
                switch textureClass {
                case "constant":
                        texture = try parameters.findRGBSpectrumTexture(name: "value")
                case "imagemap":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName)
                case "ptex":
                        let fileName = try parameters.findString(called: "filename") ?? ""
                        texture = try getTextureFrom(name: fileName)
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
                print(readTimer.elapsed)
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

        func makeMaterial(name: String, parameters: ParameterDictionary) throws -> Material {

                func makeDefault(insteadOf material: String) throws -> Material {
                        warnOnce("Unknown material \"\(material)\". Creating default.")
                        var parameterList = ParameterDictionary()
                        parameterList["Kd"] = [0.5]
                        return try createMatte(parameters: parameters)
                }

                var material: Material
                switch name {
                case "coateddiffuse":
                        material = try createCoatedDiffuse(parameters: parameters)
                case "conductor":
                        material = try createConductor(parameters: parameters)
                case "dielectric":
                        material = try createDielectric(parameters: parameters)
                case "diffuse":
                        material = try createMatte(parameters: parameters)
                case "diffusetransmission":
                        material = try createDiffuseTransmission(parameters: parameters)
                case "glass":
                        material = try createGlass(parameters: parameters)
                case "hair":
                        material = try createHair(parameters: parameters)
                case "matte":
                        material = try createMatte(parameters: parameters)
                case "metal":
                        material = try createMetal(parameters: parameters)
                case "mirror":
                        material = try createMirror(parameters: parameters)
                case "none":
                        material = try createNone()
                case "substrate":
                        material = try createSubstrate(parameters: parameters)
                default:
                        material = try makeDefault(insteadOf: name)
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
                case "curve":
                        return try createCurveShape(
                                objectToWorld: objectToWorld,
                                parameters: parameters)
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

var api = Api()
var options = Options()
var state = State()
var states = [State]()
var currentTransform = Transform()
var transforms = [Transform]()
var readTimer = Timer("")
var acceleratorName = "embree"
