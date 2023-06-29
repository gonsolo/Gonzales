final class Interface {

        func setBsdf(interaction: inout SurfaceInteraction) {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
