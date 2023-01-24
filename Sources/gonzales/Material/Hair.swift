final class Hair: Material {

        init() {}

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                var bsdf = BSDF(interaction: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: gray))
                return bsdf
        }
}

func createHair(parameters: ParameterDictionary) throws -> Hair {
        return Hair()
}
