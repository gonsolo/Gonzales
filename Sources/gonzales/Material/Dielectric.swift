final class Dielectric: Material {

        init() {}

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                // TODO
                var bsdf = BSDF(interaction: interaction)
                bsdf.set(bxdf: LambertianReflection(reflectance: white))
                return bsdf
        }
}

func createDielectric(parameters: ParameterDictionary) throws -> Conductor {
        return Conductor()
}
