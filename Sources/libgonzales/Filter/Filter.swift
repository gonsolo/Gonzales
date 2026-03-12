/// A type that evaluates a filter function at a specific location.

protocol Filter: Sendable {

        func evaluate(atLocation: Point2f) -> Real

        func sample(uSample: (Real, Real)) -> FilterSample

        var support: Vector2F { get }
}

struct FilterSample {
        let location: Point2f
        let probabilityDensity: Real
}
