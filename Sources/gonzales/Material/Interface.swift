struct Interface {

        func getBsdf() -> GlobalBsdfType {
                // Unimplemented by design
                unimplemented()
        }
}

func createInterface(parameters: ParameterDictionary) throws -> Interface {
        return Interface()
}
