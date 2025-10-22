#if os(Linux)
import Glibc
#else
import Darwin
#endif

struct GaussianFilter: Filter {

        init(withSupport support: Vector2F, withSigma sigma: FloatX) {
                self.support = support
                self.sigma = sigma
                exponent.0 = gaussian(x: support.x, sigma: sigma)
                exponent.1 = gaussian(x: support.y, sigma: sigma)
        }

        func gaussian(x: FloatX, mu: FloatX = 0, sigma: FloatX) -> FloatX {
                return 1 / (2 * FloatX.pi * sigma * sigma).squareRoot()
                        * exp(-square(x - mu) / (2 * sigma * sigma))
        }

        func evaluate(atLocation point: Point2f) -> FloatX {
                let x = max(0, gaussian(x: point.x, sigma: sigma))
                let y = max(0, gaussian(x: point.y, sigma: sigma))
                return x * y
        }

        let sigma: FloatX
        var exponent: (FloatX, FloatX) = (0, 0)
        var support: Vector2F
}
