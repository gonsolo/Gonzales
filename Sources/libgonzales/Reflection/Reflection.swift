func sameHemisphere(_ vector1: Vector, _ vector2: Vector) -> Bool { vector1.z * vector2.z > 0 }

func absCosTheta(_ v: Vector) -> FloatX { abs(v.z) }

func reflect(vector: Vector, by normal: Vector) -> Vector {
        -vector + 2 * dot(vector, normal) * normal
}

func refract(incident: Vector, normal: Normal, eta: FloatX) -> (Vector, FloatX)? {
        var eta = eta
        var cosThetaI = dot(normal, incident)
        var normal = -normal
        if cosThetaI < 0 {
                eta = 1 / eta
                cosThetaI = -cosThetaI
                normal = -normal
        }
        let sin2ThetaI = max(0, 1 - square(cosThetaI))
        let sin2ThetaT = sin2ThetaI / square(eta)
        if sin2ThetaT >= 1 {
                return nil
        }
        let cosThetaT = (1 - sin2ThetaT).squareRoot()
        let transmitted = eta * -incident + (eta * cosThetaI - cosThetaT) * Vector(normal: normal)
        return (transmitted, eta)
}

func sinTheta(_ w: Vector) -> FloatX { return sin2Theta(w).squareRoot() }
func sin2Theta(_ w: Vector) -> FloatX { return max(0, 1 - cos2Theta(w)) }
func cosTheta(_ w: Vector) -> FloatX { return w.z }
func cos2Theta(_ w: Vector) -> FloatX { return w.z * w.z }
func tan2Theta(_ w: Vector) -> FloatX { return sin2Theta(w) / cos2Theta(w) }
func tanTheta(_ w: Vector) -> FloatX { return sinTheta(w) / cosTheta(w) }

func sinPhi(_ w: Vector) -> FloatX {
        if sinTheta(w) == 0 {
                return 0
        } else {
                return clamp(value: w.y / sinTheta(w), low: -1, high: 1)
        }
}

func sin2Phi(_ w: Vector) -> FloatX { return sinPhi(w) * sinPhi(w) }
func cos2Phi(_ w: Vector) -> FloatX { return cosPhi(w) * cosPhi(w) }

func cosPhi(_ w: Vector) -> FloatX {
        if sinTheta(w) == 0 {
                return 1
        } else {
                return clamp(value: w.x / sinTheta(w), low: -1, high: 1)
        }
}
