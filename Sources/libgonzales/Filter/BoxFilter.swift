struct BoxFilter: Filter {

        func evaluate(atLocation _: Point2f) -> FloatX { return 1 }

        func sample(u: (FloatX, FloatX)) -> FilterSample {
                let f = FilterSample(
                        location: Point2f(
                                x: (2 * u.0 - 1) * support.x,
                                y: (2 * u.1 - 1) * support.y
                        ),
                        probabilityDensity: 1
                )
                return f
        }

        var support: Vector2F
}
