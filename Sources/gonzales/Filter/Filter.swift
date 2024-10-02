/// A type that evaluates a filter function at a specific location.

protocol Filter: Sendable {

        func evaluate(atLocation: Point2f) -> FloatX

        var support: Vector2F { get }
}
