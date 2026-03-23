import Foundation

protocol DefaultInitializable {
        init()
}

enum PlyError: Error {
        case noZero
        case unsupported
}

extension UInt8: DefaultInitializable {}
extension UInt32: DefaultInitializable {}
extension Int32: DefaultInitializable {}
extension Int: DefaultInitializable {}
extension Float64: DefaultInitializable {}
extension Float32: DefaultInitializable {}
#if os(Linux) && swift(>=5.4)
        extension Float16: DefaultInitializable {}
#endif
extension Point: DefaultInitializable {}
extension Normal: DefaultInitializable {}

struct PlyMesh {

        enum Endianness {
                case little
                case big
        }

        var plyHeader = PlyHeader()
        var points = [Point]()
        var normals = [Normal]()
        var uvs = [Vector2F]()
        var indices = [Int]()
        var faceIndices = [Int]()
        var hasFaceIndices = false
        var dataIndex = 0
        var listBits = 8
        var endianness = Endianness.little

        init(from data: Data) throws {
                self.init()
                try data.withUnsafeBytes { buffer in
                        try self.readPlyHeader(from: buffer)
                        try self.readVertices(from: buffer)
                        try self.readFaces(from: buffer)
                }
        }

        private init() {}
}

extension PlyMesh {

        enum PropertyType { case float, int }
        enum PropertyName { case x, y, z, normalX, normalY, normalZ, s, t, uProperty, vProperty }
        struct Property {
                let type: PropertyType
                let name: PropertyName
        }

        struct PlyHeader {
                var vertexCount = 0
                var faceCount = 0
                var vertexProperties = [Property]()
        }

        private func convert<T: DefaultInitializable>(
                buffer: UnsafeRawBufferPointer,
                at index: inout Int,
                peek: Bool = false
        ) -> T {
                let size = MemoryLayout<T>.size
                var value = buffer.load(fromByteOffset: index, as: T.self)
                if endianness == .big {
                        withUnsafeMutableBytes(of: &value) { rawBuffer in
                                var ptr = rawBuffer
                                ptr.reverse()
                        }
                }
                if !peek {
                        index += size
                }
                return value
        }

        private func readValue<T: DefaultInitializable>(
                in buffer: UnsafeRawBufferPointer, at index: inout Int
        ) -> T {
                return convert(buffer: buffer, at: &index)
        }

        private func peekValue<T: DefaultInitializable>(
                in buffer: UnsafeRawBufferPointer, at index: inout Int
        ) -> T {
                return convert(buffer: buffer, at: &index, peek: true)
        }

        private func readCharacter(in buffer: UnsafeRawBufferPointer, at index: inout Int) -> Character {
                let char = Character(UnicodeScalar(buffer[index]))
                index += 1
                return char
        }

        private func peekCharacter(in buffer: UnsafeRawBufferPointer, at index: inout Int) -> Character {
                return Character(UnicodeScalar(buffer[index]))
        }

        // swiftlint:disable:next cyclomatic_complexity function_body_length
        mutating func readPlyHeader(from buffer: UnsafeRawBufferPointer) throws {
                enum HeaderState { case vertex, face, none }
                var headerState = HeaderState.none

                guard readLine(in: buffer) == "ply" else {
                        throw SceneDescriptionError.ply(message: "First line must be ply")
                }
                while true {
                        let line = readLine(in: buffer)
                        let words = line.components(separatedBy: " ")
                        switch words[0] {
                        case "comment":
                                break
                        case "format":
                                switch words[1] {
                                case "binary_little_endian":
                                        endianness = .little
                                        guard words[2] == "1.0" else {
                                                throw SceneDescriptionError.ply(message: "1.0")
                                        }
                                case "binary_big_endian":
                                        endianness = .big
                                        guard words[2] == "1.0" else {
                                                throw SceneDescriptionError.ply(message: "1.0")
                                        }
                                default:
                                        let message = "Unknown endian format in ply."
                                        throw SceneDescriptionError.ply(message: message)
                                }
                        case "element":
                                switch words[1] {
                                case "vertex":
                                        headerState = .vertex
                                        guard let vertexCount = Int(words[2]) else {
                                                throw SceneDescriptionError.ply(message: "vertexCount")
                                        }
                                        plyHeader.vertexCount = vertexCount
                                case "face":
                                        headerState = .face
                                        guard let faceCount = Int(words[2]) else {
                                                throw SceneDescriptionError.ply(message: "faceCount")
                                        }
                                        plyHeader.faceCount = faceCount
                                default:
                                        throw SceneDescriptionError.ply(
                                                message: "Unknown element \(words[1])")
                                }
                        case "property":
                                switch words[1] {
                                case "float":
                                        guard headerState == .vertex else {
                                                throw SceneDescriptionError.ply(message: "headerState vertex")
                                        }
                                        switch words[2] {
                                        case "x":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .x))
                                        case "y":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .y))
                                        case "z":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .z))
                                        case "nx":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .normalX))
                                        case "ny":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .normalY))
                                        case "nz":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .normalZ))
                                        case "s":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .s))
                                        case "t":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .t))
                                        case "u":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .uProperty))
                                        case "v":
                                                plyHeader.vertexProperties.append(
                                                        Property(type: .float, name: .vProperty))
                                        default:
                                                throw SceneDescriptionError.ply(
                                                        message:
                                                                "Unknown float property \(words[2])"
                                                )
                                        }
                                case "list":
                                        switch words[2] {
                                        case "uint":
                                                listBits = 32
                                        case "uchar", "uint8":
                                                listBits = 8
                                        default:
                                                throw SceneDescriptionError.ply(
                                                        message:
                                                                "Unknown list property \(words[2])"
                                                )
                                        }
                                case "int":
                                        switch words[2] {
                                        case "face_indices":
                                                hasFaceIndices = true
                                        default:
                                                throw SceneDescriptionError.ply(
                                                        message: "Unknown int property \(words[2])")
                                        }
                                default:
                                        throw SceneDescriptionError.ply(
                                                message: "Unknown property: \(words[1])")
                                }
                        case "end_header":
                                return
                        default:
                                throw SceneDescriptionError.ply(message: "Unknown ply word: \"\(words[0])\"")
                        }
                }
        }

        mutating func appendPoint(from buffer: UnsafeRawBufferPointer) {
                let x: Real = readValue(in: buffer, at: &dataIndex)
                let y: Real = readValue(in: buffer, at: &dataIndex)
                let z: Real = readValue(in: buffer, at: &dataIndex)
                points.append(Point(x: Real(x), y: Real(y), z: Real(z)))
        }

        mutating func appendNormal(from buffer: UnsafeRawBufferPointer) {
                let normalX: Real = readValue(in: buffer, at: &dataIndex)
                let normalY: Real = readValue(in: buffer, at: &dataIndex)
                let normalZ: Real = readValue(in: buffer, at: &dataIndex)
                normals.append(Normal(x: Real(normalX), y: Real(normalY), z: Real(normalZ)))
        }

        mutating func appendUV(from buffer: UnsafeRawBufferPointer) {
                let uCoord: Real = readValue(in: buffer, at: &dataIndex)
                let vCoord: Real = readValue(in: buffer, at: &dataIndex)
                uvs.append(Vector2F(x: Real(uCoord), y: Real(vCoord)))
        }

        mutating func readVertices(from buffer: UnsafeRawBufferPointer) throws {
                let properties = plyHeader.vertexProperties

                points.reserveCapacity(plyHeader.vertexCount)

                for _ in 0..<plyHeader.vertexCount {

                        if plyHeader.vertexProperties.count == 3 {
                                guard
                                        properties[0].name == PropertyName.x
                                                && properties[1].name == PropertyName.y
                                                && properties[2].name == PropertyName.z
                                else { throw PlyError.unsupported }
                                appendPoint(from: buffer)
                        } else if plyHeader.vertexProperties.count == 5 {
                                guard
                                        properties[0].name == PropertyName.x
                                                && properties[1].name == PropertyName.y
                                                && properties[2].name == PropertyName.z
                                                && properties[3].name == PropertyName.uProperty
                                                && properties[4].name == PropertyName.vProperty
                                else { throw PlyError.unsupported }
                                appendPoint(from: buffer)
                                appendUV(from: buffer)
                        } else if plyHeader.vertexProperties.count == 6 {
                                guard
                                        properties[0].name == PropertyName.x
                                                && properties[1].name == PropertyName.y
                                                && properties[2].name == PropertyName.z
                                                && properties[3].name == PropertyName.normalX
                                                && properties[4].name == PropertyName.normalY
                                                && properties[5].name == PropertyName.normalZ
                                else { throw PlyError.unsupported }
                                appendPoint(from: buffer)
                                appendNormal(from: buffer)
                        } else if plyHeader.vertexProperties.count == 8 {
                                if properties[0].name == PropertyName.x
                                        && properties[1].name == PropertyName.y
                                        && properties[2].name == PropertyName.z
                                        && properties[3].name == PropertyName.normalX
                                        && properties[4].name == PropertyName.normalY
                                        && properties[5].name == PropertyName.normalZ
                                        && (properties[6].name == PropertyName.uProperty
                                                || properties[6].name == PropertyName.s)
                                        && (properties[7].name == PropertyName.vProperty
                                                || properties[7].name == PropertyName.t) {
                                        appendPoint(from: buffer)
                                        appendNormal(from: buffer)
                                        appendUV(from: buffer)
                                } else if properties[0].name == PropertyName.x
                                        && properties[1].name == PropertyName.y
                                        && properties[2].name == PropertyName.z
                                        && properties[3].name == PropertyName.uProperty
                                        && properties[4].name == PropertyName.vProperty
                                        && properties[5].name == PropertyName.normalX
                                        && properties[6].name == PropertyName.normalY
                                        && properties[7].name == PropertyName.normalZ {
                                        appendPoint(from: buffer)
                                        appendUV(from: buffer)
                                        appendNormal(from: buffer)
                                } else {
                                        throw PlyError.unsupported
                                }
                        } else {
                                throw PlyError.unsupported
                        }
                }
        }

        mutating func readFaces(from buffer: UnsafeRawBufferPointer) throws {
                indices.reserveCapacity(plyHeader.faceCount * 3)
                for _ in 0..<plyHeader.faceCount {
                        var numberIndices: UInt32 = 0
                        switch listBits {
                        case 8:
                                let value: UInt8 = readValue(in: buffer, at: &dataIndex)
                                numberIndices = UInt32(value)
                        case 32:
                                let value: UInt32 = readValue(in: buffer, at: &dataIndex)
                                numberIndices = value
                        default:
                                throw SceneDescriptionError.ply(message: "Only 8 and 32 bits supported")
                        }
                        if numberIndices != 3 {
                                print("Warning: Number of indices is not 3 but \(numberIndices)")
                        }
                        for _ in 0..<numberIndices {
                                let index: Int32 = readValue(in: buffer, at: &dataIndex)
                                indices.append(Int(index))
                        }

                        if hasFaceIndices {
                                let faceIndex: Int32 = readValue(in: buffer, at: &dataIndex)
                                faceIndices.append(Int(faceIndex))
                        }
                }
        }

        mutating func readLine(in buffer: UnsafeRawBufferPointer) -> String {
                var line = ""
                var char = readCharacter(in: buffer, at: &dataIndex)
                while !char.isNewline {
                        line.append(char)
                        char = readCharacter(in: buffer, at: &dataIndex)
                }
                return line
        }
}

extension PlyMesh {
        static func create(
                objectToWorld: Transform,
                parameters: ParameterDictionary,
                sceneDirectory: String,
                triangleMeshBuilder: TriangleMeshBuilder
        ) throws -> [ShapeType] {
                let relativeFileName = try parameters.findString(called: "filename") ?? ""
                let absoluteFileName = sceneDirectory + "/" + relativeFileName
                guard FileManager.default.fileExists(atPath: absoluteFileName) else {
                        print("Warning: Could not find ply file at: \(absoluteFileName)")
                        return []
                }
                guard let file = FileHandle(forReadingAtPath: absoluteFileName) else {
                        throw RenderError.noFileHandle
                }
                let uncompressedData = file.readDataToEndOfFile()
                var data: Data
                if absoluteFileName.hasSuffix(".gz") {
                        data = try Compression.get(data: uncompressedData)
                } else {
                        data = uncompressedData
                }
                let plyMesh = try PlyMesh(from: data)
                let meshData = MeshData(
                        indices: plyMesh.indices,
                        points: plyMesh.points,
                        normals: plyMesh.normals,
                        uvs: plyMesh.uvs,
                        faceIndices: plyMesh.faceIndices)
                return try Triangle.createMesh(
                        objectToWorld: objectToWorld,
                        meshData: meshData,
                        triangleMeshBuilder: triangleMeshBuilder)

        }
}
