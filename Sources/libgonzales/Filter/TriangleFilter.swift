import Foundation

struct TriangleFilter: Filter {

        func evaluate(atLocation point: Point2f) -> FloatX {
                return max(0, support.x - abs(point.x)) * max(0, support.y - abs(point.y))
        }

        func sample(uSample: (FloatX, FloatX)) -> FilterSample {

                let sample1D: (FloatX, FloatX) -> FloatX = { uVal, supportVal in
                        let dist = uVal < 0.5 ? sqrt(2 * uVal) - 1 : 1 - sqrt(2 * (1 - uVal))
                        return supportVal * dist
                }

                let point = Point2f(x: sample1D(uSample.0, support.x), y: sample1D(uSample.1, support.y))

                return FilterSample(location: point, probabilityDensity: evaluate(atLocation: point))
        }

        var support: Vector2F
}
