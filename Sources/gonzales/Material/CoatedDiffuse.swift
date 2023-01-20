final class CoatedDiffuse: Material {

        init() {}

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                // TODO
                var bsdf = BSDF(interaction: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: white))
                return bsdf
        }
}

func createCoatedDiffuse(parameters: ParameterDictionary) throws -> CoatedDiffuse {
        return CoatedDiffuse()
}
