func sameHemisphere(_ vector1: Vector, _ vector2: Vector) -> Bool { vector1.z * vector2.z > 0 }

func absCosTheta(_ vector: Vector) -> Real { abs(vector.z) }

func reflect(vector: Vector, by normal: Vector) -> Vector {
        -vector + 2 * dot(vector, normal) * normal
}

func refract(incident: Vector, normal: Normal, eta: Real) -> (Vector, Real)? {
        var eta = eta
        var cosThetaI = dot(normal, incident)
        var flippedNormal = Vector(normal: normal)
        if cosThetaI < 0 {
                eta = 1 / eta
                cosThetaI = -cosThetaI
                flippedNormal = -flippedNormal
        }
        let sin2ThetaI = max(0, 1 - square(cosThetaI))
        let sin2ThetaT = sin2ThetaI / square(eta)
        if sin2ThetaT >= 1 {
                return nil
        }
        let cosThetaT = (1 - sin2ThetaT).squareRoot()
        let transmitted = -incident / eta + (cosThetaI / eta - cosThetaT) * flippedNormal
        return (transmitted, eta)
}

func sinTheta(_ vector: Vector) -> Real { return sin2Theta(vector).squareRoot() }
func sin2Theta(_ vector: Vector) -> Real { return max(0, 1 - cos2Theta(vector)) }
func cosTheta(_ vector: Vector) -> Real { return vector.z }
func cos2Theta(_ vector: Vector) -> Real { return vector.z * vector.z }
func tan2Theta(_ vector: Vector) -> Real { return sin2Theta(vector) / cos2Theta(vector) }
func tanTheta(_ vector: Vector) -> Real { return sinTheta(vector) / cosTheta(vector) }

func sinPhi(_ vector: Vector) -> Real {
        if sinTheta(vector) == 0 {
                return 0
        } else {
                return clamp(value: vector.y / sinTheta(vector), low: -1, high: 1)
        }
}

func sin2Phi(_ vector: Vector) -> Real { return sinPhi(vector) * sinPhi(vector) }
func cos2Phi(_ vector: Vector) -> Real { return cosPhi(vector) * cosPhi(vector) }

func cosPhi(_ vector: Vector) -> Real {
        if sinTheta(vector) == 0 {
                return 1
        } else {
                return clamp(value: vector.x / sinTheta(vector), low: -1, high: 1)
        }
}
