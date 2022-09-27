/// A type that evaluates a filter function at a specific location.

protocol Filter {
        func evaluate(atLocation: Point2F) -> FloatX
        var support: Vector2F { get }
}
