import Foundation  // Scanner, CharacterSet

@available(OSX 10.15, *)
final class Parser {

        enum RenderStatement: String {
                case accelerator = "Accelerator"
                case areaLightSource = "AreaLightSource"
                case attributeBegin = "AttributeBegin"
                case attributeEnd = "AttributeEnd"
                case camera = "Camera"
                case concatTransform = "ConcatTransform"
                case film = "Film"
                case importFile = "Import"  // import is a keyword in Swift
                case include = "Include"
                case integrator = "Integrator"
                case lightSource = "LightSource"
                case lookAt = "LookAt"
                case makeNamedMaterial = "MakeNamedMaterial"
                case makeNamedMedium = "MakeNamedMedium"
                case material = "Material"
                case mediumInterface = "MediumInterface"
                case namedMaterial = "NamedMaterial"
                case objectBegin = "ObjectBegin"
                case objectEnd = "ObjectEnd"
                case objectInstance = "ObjectInstance"
                case pixelFilter = "PixelFilter"
                case reverseOrientation = "ReverseOrientation"
                case rotate = "Rotate"
                case sampler = "Sampler"
                case scale = "Scale"
                case shape = "Shape"
                case texture = "Texture"
                case transform = "Transform"
                case transformBegin = "TransformBegin"
                case transformEnd = "TransformEnd"
                case translate = "Translate"
                case worldBegin = "WorldBegin"
                case worldEnd = "WorldEnd"
        }

        init(fileName: String, render: Bool = true, function: String = #function) throws {
                self.scanner = try PbrtScanner(path: fileName)
                self.fileName = fileName
                self.render = render
        }

        func parse() throws {
                while !scanner.isAtEnd {
                        parseComments()
                        if scanner.isAtEnd {
                                return
                        }
                        guard
                                let buffer = scanner.scanUpToCharactersList(from: ["\n", " ", "\t"])
                        else {
                                if scanner.isAtEnd {
                                        // PBRT v4 does not use WorldEnd
                                        if render && !worldEndSeen {
                                                try api.worldEnd()
                                        }
                                        return
                                }
                                var message = "Parser error in file \(fileName)"
                                message += " at \(scanner.scanLocation)"
                                try bail(message: message)
                        }
                        try handleRenderStatement(buffer)
                }
        }

        private func bail(
                function: String = #function,
                file: String = #file,
                line: Int = #line,
                message: String = ""
        ) throws -> Never {
                throw RenderError.parser(
                        function: function,
                        file: file,
                        line: line,
                        message: message
                )
        }

        private func parseString() throws -> String {
                var string = ""
                var ok = true
                ok = try parseString(string: &string)
                if ok {
                        return string
                } else {
                        try bail()
                }
        }

        private func parseStrings() throws -> [String] {
                var strings = [String]()
                var string = ""
                var ok = true
                ok = try parseString(string: &string)
                while ok {
                        strings.append(string)
                        ok = try parseString(string: &string)
                }
                return strings
        }

        private func parseTextures() throws -> [String] {
                var textures = [String]()
                var string = ""
                var ok = true
                ok = try parseString(string: &string)
                if ok {
                        textures.append(string)
                }
                return textures
        }

        private func parseFalse() throws {
                guard let _ = scanner.scanString("f") else { try bail() }
                guard let _ = scanner.scanString("a") else { try bail() }
                guard let _ = scanner.scanString("l") else { try bail() }
                guard let _ = scanner.scanString("s") else { try bail() }
                guard let _ = scanner.scanString("e") else { try bail() }
        }

        private func parseTrue() throws {
                guard let _ = scanner.scanString("t") else { try bail() }
                guard let _ = scanner.scanString("r") else { try bail() }
                guard let _ = scanner.scanString("u") else { try bail() }
                guard let _ = scanner.scanString("e") else { try bail() }
        }

        private func parseBool() throws -> [Bool] {
                var value = ""
                let f = scanner.peekString("f")
                if f != nil {
                        try parseFalse()
                        return [false]
                }
                let t = scanner.peekString("t")
                if t != nil {
                        try parseTrue()
                        return [true]
                }
                guard try parseString(string: &value) else {
                        try bail()
                }
                switch value {
                case "true": return [true]
                case "false": return [false]
                default: try bail()
                }
        }

        private func parseString(string: inout String) throws -> Bool {
                guard scanner.scanString("\"") != nil else {
                        return false
                }
                guard let s = scanner.scanUpToString("\"") else {
                        try bail()
                }
                string = s
                guard scanner.scanString("\"") != nil else {
                        try bail()
                }
                return true
        }

        private func parseFloatXs() throws -> [FloatX] {
                var values = [FloatX]()
                while let value = try parseFloatX() {
                        values.append(value)
                }
                return values
        }

        private func parseIntegers() -> [Int] {
                var values = [Int]()
                while let value = parseInteger() {
                        values.append(value)
                }
                return values
        }

        private func parseInteger() -> Int? {
                var x: Int = 0
                guard scanner.scanInt(&x) else { return nil }
                return x
        }

        private func parseFloatX() throws -> FloatX? {
                var x: Float = 0
                guard try scanner.scanFloat(&x) else { return nil }
                return FloatX(x)
        }

        private func parseThreeFloatXs() throws -> (FloatX, FloatX, FloatX)? {
                guard let x = try parseFloatX() else { return nil }
                guard let y = try parseFloatX() else { return nil }
                guard let z = try parseFloatX() else { return nil }
                let f = (x, y, z)
                return f
        }

        private func parseNamedSpectrum() throws -> (any Spectrum)? {
                var string = ""
                if try parseString(string: &string) {
                        if let namedSpectrum = namedSpectra[string] {
                                return namedSpectrum
                        } else {
                                warning("Unknown named spectrum \(string)!")
                                warning("Returning default spectrum!")
                                return RGBSpectrum(rgb: (1, 1, 1))
                        }
                }
                return nil
        }

        private func parseRGBSpectrum() throws -> [any Spectrum] {
                var spectra = [RGBSpectrum]()
                if let namedSpectrum = try parseNamedSpectrum() {
                        return [namedSpectrum]
                }
                while let f = try parseThreeFloatXs() {
                        let spectrum = RGBSpectrum(rgb: f)
                        spectra.append(spectrum)
                }
                return spectra
        }

        private func parsePoint() throws -> Point {
                guard let f = try parseThreeFloatXs() else { try bail() }
                return Point(xyz: f)
        }

        private func parsePoints() throws -> [Point] {
                var points = [Point]()
                while let f = try parseThreeFloatXs() {
                        let point = Point(xyz: f)
                        points.append(point)
                }
                return points
        }

        private func parseVector() throws -> Vector {
                guard let f = try parseThreeFloatXs() else {
                        try bail()
                }
                return Vector(xyz: f)
        }

        private func parseVectors() throws -> [Vector] {
                var vectors = [Vector]()
                while let f = try parseThreeFloatXs() {
                        let vector = Vector(xyz: f)
                        vectors.append(vector)
                }
                return vectors
        }

        private func parseNormals() throws -> [Normal] {
                var normals = [Normal]()
                while let f = try parseThreeFloatXs() {
                        let normal = Normal(xyz: f)
                        normals.append(normal)
                }
                return normals
        }

        private func parseValue(type: String) throws -> Parameter {
                switch type {
                case "bool":
                        return try parseBool()
                case "float":
                        guard let value = try parseFloatX() else {
                                try bail(message: "Float expected")
                        }
                        return [value]
                case "integer":
                        guard let value = parseInteger() else {
                                try bail(message: "Int expected")
                        }
                        return [value]
                case "normal":
                        return try parseNormals()
                case "point", "point3":
                        return try parsePoints()
                case "point2":
                        // TODO: Parse as Point2f
                        return try parseFloatXs()
                case "rgb", "color", "spectrum":
                        return try parseRGBSpectrum()
                case "string":
                        let string = try parseString()
                        return [string]
                case "texture":
                        return try parseTextures()
                default:
                        var message = "Unknown type \(type)"
                        message += " at location \(scanner.scanLocation)"
                        try bail(message: message)
                }
        }

        private func parseValues(type: String) throws -> Parameter {
                switch type {
                case "bool":
                        return try parseBool()
                case "float":
                        return try parseFloatXs()
                case "integer":
                        return parseIntegers()
                case "normal":
                        return try parseNormals()
                case "point", "point3":
                        return try parsePoints()
                case "point2":
                        // TODO: Parse as Point2f
                        return try parseFloatXs()
                case "rgb", "color", "spectrum":
                        return try parseRGBSpectrum()
                case "string":
                        return try parseStrings()
                case "texture":
                        return try parseTextures()
                default:
                        var message = "Unknown type \(type)"
                        message += " at location \(scanner.scanLocation)"
                        try bail(message: message)
                }
        }

        private func parseParameter() throws -> (String, Parameter)? {
                parseComments()
                guard let _ = scanner.scanString("\"") else { return nil }
                guard let type = scanner.scanUpToCharactersList(from: ["\n", " "]) else {
                        try bail()
                }
                guard let name = scanner.scanUpToCharactersList(from: ["\""]) else {
                        try bail()
                }
                guard let _ = scanner.scanString("\"") else {
                        try bail()
                }
                let singleValue = scanner.scanString("[") == nil
                let parameter = singleValue ? try parseValue(type: type) : try parseValues(type: type)
                if !singleValue {
                        _ = scanner.scanString("]")
                }
                return (name, parameter)
        }

        private func parseParameters() throws -> ParameterDictionary {
                var parameters = ParameterDictionary()
                var nameAndParameter = try parseParameter()
                while nameAndParameter != nil {
                        let name = nameAndParameter!.0
                        let parameter = nameAndParameter!.1
                        parameters[name] = parameter
                        parseComments()  // TODO: This does not belong here!
                        nameAndParameter = try parseParameter()
                }
                return parameters
        }

        private func parseIntegrator() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.integrator(name: name, parameters: parameters)
        }

        private func parseLightSource() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.lightSource(name: name, parameters: parameters)
        }

        private func parseLookAt() throws {
                let eye = try parsePoint()
                let at = try parsePoint()
                let up = try parseVector()
                try api.lookAt(eye: eye, at: at, up: up)
        }

        private func parseRotate() throws {
                guard let angle = try parseFloatX() else { try bail() }
                guard let f = try parseThreeFloatXs() else { try bail() }
                let axis = Vector(xyz: f)
                try api.rotate(by: FloatX(angle), around: axis)
        }

        private func scanTransform() throws -> [FloatX] {
                guard let _ = scanner.scanString("[") else { try bail() }
                var values = [FloatX]()
                for _ in 0..<16 {
                        guard let value = try parseFloatX() else { try bail() }
                        values.append(value)
                }
                guard let _ = scanner.scanString("]") else { try bail() }
                return values
        }

        private func parseTransform() throws {
                let values = try scanTransform()
                try api.transform(values: values)
        }

        private func parseTransformBegin() throws {
                try api.transformBegin()
        }

        private func parseTransformEnd() throws {
                try api.transformEnd()
        }

        private func parseTranslate() throws {
                guard let f = try parseThreeFloatXs() else { try bail() }
                let translation = Vector(xyz: f)
                try api.translate(by: translation)
        }

        private func parseSampler() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                api.sampler(name: name, parameters: parameters)
        }

        private func parseScale() throws {
                guard let f = try parseThreeFloatXs() else { try bail() }
                try api.scale(x: f.0, y: f.1, z: f.2)
        }

        private func parsePixelFilter() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                api.pixelFilter(name: name, parameters: parameters)
        }

        private func parseReverseOrientation() {
                warning("Ignoring reverseOrientation!")
        }

        private func parseFilm() throws {
                let name = try parseString()
                switch name {
                case "image":
                        break
                case "rgb":  // v4
                        break
                default:
                        try bail()
                }
                let parameters = try parseParameters()
                try api.film(name: name, parameters: parameters)
        }

        private func parseImport() throws {
                let name = try parseString()
                try api.importFile(file: name)
        }

        private func parseInclude() throws {
                let name = try parseString()
                try api.include(file: name, render: false)
        }

        private func parseCamera() throws {
                let name = try parseString()
                guard name == "perspective" else {
                        try bail()
                }
                let parameters = try parseParameters()
                try api.camera(name: name, parameters: parameters)
        }

        private func parseConcatTransform() throws {
                let values = try scanTransform()
                try api.concatTransform(values: values)
        }

        private func parseWorldBegin() throws {
                api.worldBegin()
        }

        private func parseWorldEnd() throws {
                worldEndSeen = true
                try api.worldEnd()
        }

        private func parseMakeNamedMaterial() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.makeNamedMaterial(name: name, parameters: parameters)
        }

        private func parseMakeNamedMedium() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.makeNamedMedium(name: name, parameters: parameters)
        }

        private func parseMaterial() throws {
                let type = try parseString()
                let parameters = try parseParameters()
                try api.material(type: type, parameters: parameters)
        }

        private func parseMediumInterface() throws {
                let interior = try parseString()
                let exterior = try parseString()
                api.mediumInterface(interior: interior, exterior: exterior)
        }

        private func parseNamedMaterial() throws {
                let name = try parseString()
                try api.namedMaterial(name: name)
        }

        private func parseObjectBegin() throws {
                let name = try parseString()
                try api.objectBegin(name: name)
        }

        private func parseObjectEnd() throws {
                try api.objectEnd()
        }

        private func parseObjectInstance() throws {
                let name = try parseString()
                try api.objectInstance(name: name)
        }

        private func parseShape() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.shape(name: name, parameters: parameters)
        }

        private func parseTexture() throws {
                let textureName = try parseString()
                let textureType = try parseString()
                let textureClass = try parseString()
                let parameters = try parseParameters()
                try api.texture(
                        name: textureName,
                        type: textureType,
                        textureClass: textureClass,
                        parameters: parameters
                )
        }

        private func parseAttributeBegin() throws {
                try api.attributeBegin()
        }

        private func parseAttributeEnd() throws {
                try api.attributeEnd()
        }

        private func parseAccelerator() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.accelerator(name: name, parameters: parameters)
        }

        private func parseAreaLightSource() throws {
                let name = try parseString()
                let parameters = try parseParameters()
                try api.areaLight(name: name, parameters: parameters)
        }

        private func parseComment() -> Bool {
                let hashmark = scanner.peekString("#")
                if hashmark != nil {
                        _ = scanner.scanUpToCharactersList(from: ["\n"])
                        return true
                }
                return false
        }

        private func parseComments() {
                var commentFound: Bool
                repeat {
                        commentFound = parseComment()
                } while commentFound
        }

        private func handleRenderStatement(_ input: String) throws {
                guard let statement = RenderStatement(rawValue: input) else {
                        var message = "Unknown RenderStatement: |\(input)|"
                        message += " in file \(fileName)"
                        message += " at location \(scanner.scanLocation)"
                        try bail(message: message)
                }
                switch statement {
                case .accelerator: try parseAccelerator()
                case .areaLightSource: try parseAreaLightSource()
                case .attributeBegin: try parseAttributeBegin()
                case .attributeEnd: try parseAttributeEnd()
                case .camera: try parseCamera()
                case .concatTransform: try parseConcatTransform()
                case .film: try parseFilm()
                case .include: try parseInclude()
                case .importFile: try parseImport()
                case .integrator: try parseIntegrator()
                case .lightSource: try parseLightSource()
                case .lookAt: try parseLookAt()
                case .makeNamedMaterial: try parseMakeNamedMaterial()
                case .makeNamedMedium: try parseMakeNamedMedium()
                case .material: try parseMaterial()
                case .mediumInterface: try parseMediumInterface()
                case .namedMaterial: try parseNamedMaterial()
                case .objectBegin: try parseObjectBegin()
                case .objectEnd: try parseObjectEnd()
                case .objectInstance: try parseObjectInstance()
                case .pixelFilter: try parsePixelFilter()
                case .reverseOrientation: parseReverseOrientation()
                case .rotate: try parseRotate()
                case .sampler: try parseSampler()
                case .scale: try parseScale()
                case .shape: try parseShape()
                case .texture: try parseTexture()
                case .transform: try parseTransform()
                case .transformBegin: try parseTransformBegin()
                case .transformEnd: try parseTransformEnd()
                case .translate: try parseTranslate()
                case .worldBegin: try parseWorldBegin()
                case .worldEnd: try parseWorldEnd()
                }
        }

        var scanner: PbrtScanner
        let fileName: String
        var render = true
        var worldEndSeen = false
}
