struct Interface {

        func getBsdf() -> GlobalBsdf {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
