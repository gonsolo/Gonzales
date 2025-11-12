enum MeasuredError: Error {
        case emptyFileName
}

struct Measured {

        func getBsdf(interaction: any Interaction) -> DiffuseBsdf {
                unimplemented()
        }
}

func createMeasured(parameters _: ParameterDictionary) throws -> Measured {
        return Measured()
}
