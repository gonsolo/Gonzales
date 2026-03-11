enum MeasuredError: Error {
        case emptyFileName
}

struct Measured {

        func getBsdf(_: any Interaction) -> DiffuseBsdf {
                unimplemented()
        }
}

extension Measured {
        static func create(parameters _: ParameterDictionary) throws -> Measured {
        return Measured()
}
}
