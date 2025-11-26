public enum ParameterError: Error {
        case missing(parameter: String, function: String)
        case isNil
}
