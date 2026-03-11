struct Interface {

        func getBsdf() -> GlobalBsdfType {
                // Unimplemented by design
                unimplemented()
        }
}

extension Interface {
        static func create(parameters _: ParameterDictionary) throws -> Interface {
        return Interface()
}
}
