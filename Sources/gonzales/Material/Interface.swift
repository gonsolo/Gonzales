final class Interface: Material {

        func getGlobalBsdf(interaction: Interaction) -> GlobalBsdf {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
