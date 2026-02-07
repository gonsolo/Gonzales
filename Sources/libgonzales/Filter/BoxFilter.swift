struct BoxFilter: Filter {

        func evaluate(atLocation _: Point2f) -> FloatX { return 1 }

        func sample(uSample: (FloatX, FloatX)) -> FilterSample {
                let sample = FilterSample(
                        location: Point2f(
                                x: (2 * uSample.0 - 1) * support.x,
                                y: (2 * uSample.1 - 1) * support.y
                        ),
                        probabilityDensity: 1
                )
                return sample
        }

        var support: Vector2F
}
