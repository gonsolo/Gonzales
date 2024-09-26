import Glibc

enum RenderError: Error {
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
}

func unimplemented(
        function: String = #function, file: String = #file, line: Int = #line, message: String = ""
) -> Never {
        print("Unimplemented in function \(function) in file \(file), line \(line), \(message)")
        fatalError()
}

func todo(
        function: String = #function, file: String = #file, line: Int = #line, message: String = ""
) -> Never {
        unimplemented(function: function, file: file, line: line, message: message)
}

func notOverridden(function: String = #function, file: String = #file, line: Int = #line) -> Never {
        print("Function \(function) in file \(file), line \(line) has to be overridden!")
        fatalError()
}

@MainActor
func warning(_ message: String) {
        if verbose {
                print("Warning: \(message)")
        }
}

@MainActor
var warningsSeen = Set<Int>()

@MainActor
func warnOnce(_ message: String) {
        var hasher = Hasher()
        hasher.combine(message)
        let hash = hasher.finalize()
        if !warningsSeen.contains(hash) {
                warningsSeen.insert(hash)
                warning(message)
        }
}

func abort(_ message: String) -> Never {
        print("Error: \(message)")
        exit(-1)
}
