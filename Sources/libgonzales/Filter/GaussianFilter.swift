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

        func gaussian(x: FloatX, mean: FloatX = 0, sigma: FloatX) -> FloatX {
                return 1 / (2 * FloatX.pi * sigma * sigma).squareRoot()
                        * exp(-square(x - mean) / (2 * sigma * sigma))
        }

        func evaluate(atLocation point: Point2f) -> FloatX {
                let x = max(0, gaussian(x: point.x, sigma: sigma))
                let y = max(0, gaussian(x: point.y, sigma: sigma))
                return x * y
        }

        func erfinv(y: FloatX) -> FloatX {
                let center: FloatX = 0.7

                let coeffA: [FloatX] = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
                let coeffB: [FloatX] = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
                let coeffC: [FloatX] = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
                let coeffD: [FloatX] = [3.543889200, 1.637067800]

                let absY = abs(y)

                if absY <= center {
                        let z = y * y

                        var numPoly = coeffA[3] * z + coeffA[2]
                        numPoly = numPoly * z + coeffA[1]
                        let num = numPoly * z + coeffA[0]

                        var denPoly = coeffB[3] * z + coeffB[2]
                        denPoly = denPoly * z + coeffB[1]
                        denPoly = denPoly * z + coeffB[0]
                        let den = denPoly * z + 1.0

                        var x = y * num / den

                        let piValue = FloatX.pi
                        let twoOverSqrtPi = 2.0 / sqrt(piValue)

                        let updateTerm: (FloatX) -> FloatX = { currentX in
                                (erf(currentX) - y) / (twoOverSqrtPi * exp(-(currentX * currentX)))
                        }

                        x -= updateTerm(x)
                        x -= updateTerm(x)

                        return x
                } else if absY < 1.0 {

                        let zBase: FloatX = (1.0 - absY) / 2.0
                        let z: FloatX = sqrt(-log(zBase))

                        var numPoly = coeffC[3] * z + coeffC[2]
                        numPoly = numPoly * z + coeffC[1]
                        let num = numPoly * z + coeffC[0]

                        let denPoly = coeffD[1] * z + coeffD[0]
                        let den = denPoly * z + 1.0

                        let signY: FloatX = y.sign == .plus ? 1.0 : -1.0

                        var x = signY * num / den

                        let piValue = FloatX.pi
                        let twoOverSqrtPi = 2.0 / sqrt(piValue)

                        let updateTerm: (FloatX) -> FloatX = { currentX in
                                (erf(currentX) - y) / (twoOverSqrtPi * exp(-(currentX * currentX)))
                        }

                        x -= updateTerm(x)
                        x -= updateTerm(x)

                        return x
                } else if absY == 1.0 {
                        return y.sign == .plus ? .infinity : -.infinity
                } else {
                        return .nan
                }
        }

        func sample(uSample: (FloatX, FloatX)) -> FilterSample {

                func sample1D(uSample: FloatX, radius: FloatX, sigma: FloatX) -> (sample: FloatX, density: FloatX) {
                        let norm = 0.5 * (1 + erf(radius / (sigma * sqrt(2))))

                        let uScaled = (1 - norm) + uSample * (2 * norm - 1)

                        let x = sigma * sqrt(2) * erfinv(y: 2 * uScaled - 1)

                        let pdfUnnormalized =
                                exp(-x * x / (2 * sigma * sigma)) / (sigma * sqrt(2 * FloatX.pi))

                        let density = pdfUnnormalized / (2 * norm - 1)

                        let clampedX = max(-radius, min(radius, x))

                        return (sample: clampedX, density: density)
                }

                let resultX = sample1D(uSample: uSample.0, radius: support.x, sigma: sigma)
                let resultY = sample1D(uSample: uSample.1, radius: support.y, sigma: sigma)
                let location = Point2f(x: resultX.sample, y: resultY.sample)
                let probabilityDensity = resultX.density * resultY.density

                return FilterSample(location: location, probabilityDensity: probabilityDensity)
        }

        let sigma: FloatX
        var exponent: (FloatX, FloatX) = (0, 0)
        var support: Vector2F
}
