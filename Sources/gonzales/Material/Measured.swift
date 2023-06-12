enum MeasuredError: Error {
        case emptyFileName
}

final class Measured: Material {

        func getGlobalBsdf(interaction: Interaction) -> GlobalBsdf {
                // TODO: Implement this some day
                let bsdfGeometry = BsdfGeometry(interaction: interaction)
                let diffuseBsdf = DiffuseBsdf(bsdfGeometry: bsdfGeometry)
                return diffuseBsdf
        }
}

func createMeasured(parameters: ParameterDictionary) throws -> Measured {
        return Measured()
}
