/// A type that can be bound in Euclidaen space.
///
/// Typically this is used to restrict intersection calculations to a
/// subset of all primitives.

protocol Boundable: Sendable {
        func objectBound(scene: Scene) async -> Bounds3f
        func worldBound(scene: Scene) async -> Bounds3f
}
