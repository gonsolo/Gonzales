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

        func erfinv(y: FloatX) -> FloatX {
                let center: FloatX = 0.7

                let a: [FloatX] = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
                let b: [FloatX] = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
                let c: [FloatX] = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
                let d: [FloatX] = [3.543889200, 1.637067800]

                let abs_y = abs(y)

                if abs_y <= center {
                        let z = y * y

                        var num_poly = a[3] * z + a[2]
                        num_poly = num_poly * z + a[1]
                        let num = num_poly * z + a[0]

                        var den_poly = b[3] * z + b[2]
                        den_poly = den_poly * z + b[1]
                        den_poly = den_poly * z + b[0]
                        let den = den_poly * z + 1.0

                        var x = y * num / den

                        let pi = FloatX.pi
                        let two_over_sqrt_pi = 2.0 / sqrt(pi)

                        let updateTerm: (FloatX) -> FloatX = { current_x in
                                (erf(current_x) - y) / (two_over_sqrt_pi * exp(-(current_x * current_x)))
                        }

                        x = x - updateTerm(x)
                        x = x - updateTerm(x)

                        return x
                }

                else if abs_y < 1.0 {

                        let z_base: FloatX = (1.0 - abs_y) / 2.0
                        let z: FloatX = sqrt(-log(z_base))

                        var num_poly = c[3] * z + c[2]
                        num_poly = num_poly * z + c[1]
                        let num = num_poly * z + c[0]

                        let den_poly = d[1] * z + d[0]
                        let den = den_poly * z + 1.0

                        let sign_y: FloatX = y.sign == .plus ? 1.0 : -1.0

                        var x = sign_y * num / den

                        let pi = FloatX.pi
                        let two_over_sqrt_pi = 2.0 / sqrt(pi)

                        let updateTerm: (FloatX) -> FloatX = { current_x in
                                (erf(current_x) - y) / (two_over_sqrt_pi * exp(-(current_x * current_x)))
                        }

                        x = x - updateTerm(x)
                        x = x - updateTerm(x)

                        return x
                }

                else if abs_y == 1.0 {
                        return y.sign == .plus ? .infinity : -.infinity
                } else {
                        return .nan
                }
        }

        func sample(u: (FloatX, FloatX)) -> FilterSample {

                let u_x = u.0
                let u_y = u.1

                func sample1D(u: FloatX, radius: FloatX, sigma: FloatX) -> (sample: FloatX, density: FloatX) {
                        let norm = 0.5 * (1 + erf(radius / (sigma * sqrt(2))))

                        let uScaled = (1 - norm) + u * (2 * norm - 1)

                        let x = sigma * sqrt(2) * erfinv(y: 2 * uScaled - 1)

                        let pdf_unnormalized =
                                exp(-x * x / (2 * sigma * sigma)) / (sigma * sqrt(2 * FloatX.pi))

                        let density = pdf_unnormalized / (2 * norm - 1)

                        let clamped_x = max(-radius, min(radius, x))

                        return (sample: clamped_x, density: density)
                }

                let result_x = sample1D(u: u_x, radius: support.x, sigma: sigma)
                let result_y = sample1D(u: u_y, radius: support.y, sigma: sigma)
                let location = Point2f(x: result_x.sample, y: result_y.sample)
                let probabilityDensity = result_x.density * result_y.density

                return FilterSample(location: location, probabilityDensity: probabilityDensity)
        }

        let sigma: FloatX
        var exponent: (FloatX, FloatX) = (0, 0)
        var support: Vector2F
}
