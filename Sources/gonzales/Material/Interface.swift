final class Interface: Material {

        func getBSDF(interaction: Interaction) -> BSDF {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
