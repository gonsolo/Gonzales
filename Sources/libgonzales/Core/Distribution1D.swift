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
                for index in 1...count {
                        cdf[index] = cdf[index - 1] + function[index - 1] / Real(count)
                }
                self.integral = cdf[count]
                if self.integral == 0 {
                        for index in 1...count {
                                cdf[index] = Real(index) / Real(count)
                        }
                } else {
                        for index in 1...count {
                                cdf[index] /= self.integral
                        }
                }
                self.cdf = cdf
        }

        func countElements() -> Int {
                return count
        }

        func findInterval(sample: Real) -> Int {
                var low = 0
                var high = count
                while low < high {
                        let mid = low + (high - low) / 2
                        if cdf[mid] <= sample {
                                low = mid + 1
                        } else {
                                high = mid
                        }
                }
                return clamp(value: low - 1, low: 0, high: count - 1)
        }

        func sampleContinuous(sample: Real) -> (value: Real, pdf: Real, offset: Int) {
                let offset = findInterval(sample: sample)
                var delta = sample - cdf[offset]
                let distance = cdf[offset + 1] - cdf[offset]
                if distance > 0 {
                        delta /= distance
                }
                let value = (Real(offset) + delta) / Real(count)
                let probabilityDensity = pdf(value: value)
                return (value, probabilityDensity, offset)
        }

        func sampleDiscrete(sample: Real) -> (offset: Int, pdf: Real, uRemapped: Real) {
                let offset = findInterval(sample: sample)
                let probabilityDensity = function[offset] / (integral * Real(count))
                let distance = cdf[offset + 1] - cdf[offset]
                var uRemapped = (sample - cdf[offset]) / distance
                if distance == 0 || uRemapped.isNaN { uRemapped = 0 }
                return (offset, probabilityDensity, uRemapped)
        }

        func pdf(value: Real) -> Real {
                let offset = clamp(value: Int(value * Real(count)), low: 0, high: count - 1)
                return function[offset] / integral
        }

        func discretePdf(offset: Int) -> Real {
                return function[offset] / (integral * Real(count))
        }
}
