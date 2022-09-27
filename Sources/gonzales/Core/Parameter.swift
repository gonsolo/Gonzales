/// A parameter in a pbrt file.
///
/// This is just a placeholder for integer, float, etc. values
/// and arrays of integer, float and other values.

protocol Parameter {}

// Every Array can be used as a parameter.
// This is possibly too wide.

extension Array: Parameter {}
