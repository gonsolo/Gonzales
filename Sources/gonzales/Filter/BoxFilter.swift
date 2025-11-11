struct BoxFilter: Filter {

        func evaluate(atLocation: Point2f) -> FloatX { return 1 }

        func sample(u: (FloatX, FloatX)) -> FilterSample {
                unimplemented()
        }

        var support: Vector2F
}
