import Foundation

func concentricSampleDisk(uSample: TwoRandomVariables) -> Point2f {
        let uSampleVector = Point2f(x: uSample.0, y: uSample.1)
        let uOffset = 2.0 * uSampleVector - Vector2F(x: 1, y: 1)
        if uOffset.x == 0 && uOffset.y == 0 {
                return Point2f()
        }
        var theta: FloatX = 0.0
        var radius: FloatX = 0.0
        if abs(uOffset.x) > abs(uOffset.y) {
                radius = uOffset.x
                theta = (FloatX.pi / 4.0) * (uOffset.y / uOffset.x)
        } else {
                radius = uOffset.y
                theta = (FloatX.pi / 2.0) - (FloatX.pi / 4.0) * (uOffset.x / uOffset.y)
        }
        return radius * Point2f(x: cos(theta), y: sin(theta))
}

func cosineSampleHemisphere(uSample: TwoRandomVariables) -> Vector {
        let diskSample = concentricSampleDisk(uSample: uSample)
        let zComponent = sqrt(max(0, 1 - diskSample.x * diskSample.x - diskSample.y * diskSample.y))
        return Vector(x: diskSample.x, y: diskSample.y, z: zComponent)
}

func powerHeuristic(pdfF: FloatX, pdfG: FloatX) -> FloatX {
        if pdfF == 0 || pdfG == 0 { return 0 }
        return (pdfF * pdfF) / (pdfF * pdfF + pdfG * pdfG)
}
