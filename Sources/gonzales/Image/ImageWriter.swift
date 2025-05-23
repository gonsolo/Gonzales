///        A type that writes an image to a file.

enum ImageWriter: Sendable {

        case ascii(AsciiWriter)
        //case openImageIO(OpenImageIOWriter)

        func write(fileName: String, crop: Bounds2i, image: Image) async throws {
                switch self {
                case .ascii(let asciiWriter):
                        return try await asciiWriter.write(fileName: fileName, crop: crop, image: image)
                //case .openImageIO(let openImageIOWriter):
                //        return try await openImageIOWriter.write(fileName: fileName, crop: crop, image: image)
                }
        }
}
