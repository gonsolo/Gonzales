enum ParameterError: Error {
        case missing(parameter: String)
        case isNil
}
