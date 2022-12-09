final class None: Material {

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                return BSDF(interaction: interaction)
        }
}

func createNone() throws -> None {
        return None()
}
