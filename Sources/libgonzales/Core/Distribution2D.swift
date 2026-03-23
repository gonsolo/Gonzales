struct PiecewiseConstant2D {
        let pConditionalV: [PiecewiseConstant1D]
        let pMarginal: PiecewiseConstant1D

        init(data: [Real], width: Int, height: Int) {
                var conditionalV = [PiecewiseConstant1D]()
                for row in 0..<height {
                        let startIndex = row * width
                        let endIndex = startIndex + width
                        let rowData = Array(data[startIndex..<endIndex])
                        conditionalV.append(PiecewiseConstant1D(values: rowData))
                }
                self.pConditionalV = conditionalV

                var marginalFunc = [Real]()
                for row in 0..<height {
                        marginalFunc.append(conditionalV[row].integral)
                }
                self.pMarginal = PiecewiseConstant1D(values: marginalFunc)
        }

        func sampleContinuous(sample: Point2f) -> (uv: Point2f, pdf: Real) {
                let (marginalValue, pdf1, marginalOffset) = pMarginal.sampleContinuous(sample: sample.y)
                let (conditionalValue, pdf0, _) = pConditionalV[marginalOffset].sampleContinuous(
                        sample: sample.x)
                let pdf = pdf0 * pdf1
                return (Point2f(x: conditionalValue, y: marginalValue), pdf)
        }

        func pdf(texCoord: Point2f) -> Real {
                let columnIndex = clamp(
                        value: Int(texCoord.x * Real(pConditionalV[0].count)),
                        low: 0, high: pConditionalV[0].count - 1)
                let rowIndex = clamp(
                        value: Int(texCoord.y * Real(pMarginal.count)),
                        low: 0, high: pMarginal.count - 1)
                return pConditionalV[rowIndex].function[columnIndex] / pMarginal.integral
        }
}
