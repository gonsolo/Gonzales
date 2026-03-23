enum MeasuredError: Error {
        case emptyFileName
}

struct Measured {

        // TODO: Not properly implemented. Should load and evaluate measured BRDF data.
        // Falls back to white diffuse for now.
        func getBsdf(_ interaction: any Interaction) -> DiffuseBsdf {
                let bsdfFrame = BsdfFrame(interaction: interaction)
                return DiffuseBsdf(reflectance: white, bsdfFrame: bsdfFrame)
        }
}

extension Measured {
        static func create(parameters _: ParameterDictionary) throws -> Measured {
                return Measured()
        }
}
