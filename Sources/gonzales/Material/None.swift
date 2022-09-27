final class None: Material {

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                let bsdf = BSDF(interaction: interaction)
                return (bsdf, nil)
        }
}

func createNone() throws -> None {
        return None()
}
