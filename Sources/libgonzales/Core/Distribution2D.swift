struct PiecewiseConstant2D {
        let pConditionalV: [PiecewiseConstant1D]
        let pMarginal: PiecewiseConstant1D

        init(data: [Real], width: Int, height: Int) {
                var conditionalV = [PiecewiseConstant1D]()
                for v in 0..<height {
                        let startIndex = v * width
                        let endIndex = startIndex + width
                        let row = Array(data[startIndex..<endIndex])
                        conditionalV.append(PiecewiseConstant1D(values: row))
                }
                self.pConditionalV = conditionalV

                var marginalFunc = [Real]()
                for v in 0..<height {
                        marginalFunc.append(conditionalV[v].integral)
                }
                self.pMarginal = PiecewiseConstant1D(values: marginalFunc)
        }

        func sampleContinuous(u: Point2f) -> (uv: Point2f, pdf: Real) {
                let (d1, pdf1, v) = pMarginal.sampleContinuous(u: u.y)
                let (d0, pdf0, _) = pConditionalV[v].sampleContinuous(u: u.x)
                let pdf = pdf0 * pdf1
                return (Point2f(x: d0, y: d1), pdf)
        }

        func pdf(uv: Point2f) -> Real {
                let iu = clamp(value: Int(uv.x * Real(pConditionalV[0].count)), low: 0, high: pConditionalV[0].count - 1)
                let iv = clamp(value: Int(uv.y * Real(pMarginal.count)), low: 0, high: pMarginal.count - 1)
                return pConditionalV[iv].function[iu] / pMarginal.integral
        }
}
