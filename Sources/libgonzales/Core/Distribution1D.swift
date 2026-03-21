struct PiecewiseConstant1D {
        let function: [Real]
        let cdf: [Real]
        let integral: Real
        let count: Int

        init(values: [Real]) {
                self.count = values.count
                self.function = values
                var cdf = [Real](repeating: 0, count: count + 1)
                cdf[0] = 0
                for i in 1...count {
                        cdf[i] = cdf[i - 1] + function[i - 1] / Real(count)
                }
                self.integral = cdf[count]
                if self.integral == 0 {
                        for i in 1...count {
                                cdf[i] = Real(i) / Real(count)
                        }
                } else {
                        for i in 1...count {
                                cdf[i] /= self.integral
                        }
                }
                self.cdf = cdf
        }

        func countElements() -> Int {
                return count
        }

        func findInterval(u: Real) -> Int {
                var low = 0
                var high = count
                while low < high {
                        let mid = low + (high - low) / 2
                        if cdf[mid] <= u {
                                low = mid + 1
                        } else {
                                high = mid
                        }
                }
                return clamp(value: low - 1, low: 0, high: count - 1)
        }

        func sampleContinuous(u: Real) -> (value: Real, pdf: Real, offset: Int) {
                let offset = findInterval(u: u)
                var du = u - cdf[offset]
                let distance = cdf[offset + 1] - cdf[offset]
                if distance > 0 {
                        du /= distance
                }
                let value = (Real(offset) + du) / Real(count)
                let p = pdf(value: value)
                return (value, p, offset)
        }

        func sampleDiscrete(u: Real) -> (offset: Int, pdf: Real, uRemapped: Real) {
                let offset = findInterval(u: u)
                let p = function[offset] / (integral * Real(count))
                let distance = cdf[offset + 1] - cdf[offset]
                var uRemapped = (u - cdf[offset]) / distance
                if distance == 0 || uRemapped.isNaN { uRemapped = 0 }
                return (offset, p, uRemapped)
        }

        func pdf(value: Real) -> Real {
                let offset = clamp(value: Int(value * Real(count)), low: 0, high: count - 1)
                return function[offset] / integral
        }

        func discretePdf(offset: Int) -> Real {
                return function[offset] / (integral * Real(count))
        }
}
