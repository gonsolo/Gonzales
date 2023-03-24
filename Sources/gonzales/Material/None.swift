final class None: Material {

        func getBSDF(interaction: Interaction) -> BSDF {
                return BSDF(interaction: interaction)
        }
}

func createNone() throws -> None {
        return None()
}
