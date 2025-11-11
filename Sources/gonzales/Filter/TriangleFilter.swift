struct TriangleFilter: Filter {

        func evaluate(atLocation point: Point2f) -> FloatX {
                return max(0, support.x - abs(point.x)) * max(0, support.y - abs(point.y))
        }

        func sample(u: (FloatX, FloatX)) -> FilterSample {
                unimplemented()
        }

        var support: Vector2F
}
