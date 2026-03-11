public enum RenderError: Error {
        case fopen
        case insufficientArguments
        case fileNotExisting(name: String)
        case noFileHandle
        case readFile
        case parser(function: String, file: String, line: Int, message: String)
        case writeImage
        case index
        case union
        case unsupportedFilter
        case noSceneSpecified
        case noLights
        case readFloatX
        case readInt32
        case readUInt8
        case unimplemented(function: String, file: String, line: Int, message: String)
}
