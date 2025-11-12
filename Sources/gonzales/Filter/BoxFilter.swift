struct BoxFilter: Filter {

        func evaluate(atLocation _: Point2f) -> FloatX { return 1 }

        func sample(u _: (FloatX, FloatX)) -> FilterSample {
                unimplemented()
        }

        var support: Vector2F
}
