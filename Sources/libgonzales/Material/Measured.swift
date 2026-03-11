enum MeasuredError: Error {
        case emptyFileName
}

struct Measured {

        func getBsdf(_: any Interaction) throws -> DiffuseBsdf {
                throw RenderError.unimplemented(function: #function, file: #file, line: #line, message: "")
        }
}

extension Measured {
        static func create(parameters _: ParameterDictionary) throws -> Measured {
        return Measured()
}
}
