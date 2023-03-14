final class Interface: Material {

        func computeScatteringFunctions(interaction: Interaction) -> BSDF {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
