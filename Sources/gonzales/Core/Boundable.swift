/// A type that can be bound in Euclidaen space.
///
/// Typically this is used to restrict intersection calculations to a
/// subset of all primitives.

protocol Boundable {
        func objectBound() async -> Bounds3f
        func worldBound() async -> Bounds3f
}
