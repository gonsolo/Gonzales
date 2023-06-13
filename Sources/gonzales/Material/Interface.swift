final class Interface: Material {

        func getGlobalBsdf(interaction: SurfaceInteraction) -> GlobalBsdf {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
