struct Interface {

        func getBsdf() throws -> BsdfVariant {
                // Unimplemented by design
                throw RenderError.unimplemented(function: #function, file: #file, line: #line, message: "")
        }
}

extension Interface {
        static func create(parameters _: ParameterDictionary) throws -> Interface {
        return Interface()
}
}
