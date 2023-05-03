enum MeasuredError: Error {
        case emptyFileName
}

final class Measured: Material {

        func getBSDF(interaction: Interaction) -> BSDF {
                unimplemented()
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        guard let fileName = try parameters.findString(called: "filename") else {
                throw MeasuredError.emptyFileName
        }
        print("measured: ", fileName)
        return Measured()
}
