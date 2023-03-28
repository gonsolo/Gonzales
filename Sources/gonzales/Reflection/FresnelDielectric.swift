struct FresnelDielectric: Fresnel {

        func evaluate(cosThetaI: FloatX, refractiveIndex: FloatX) -> FloatX {
                var refractiveIndex = refractiveIndex
                var cosThetaI = clamp(value: cosThetaI, low: -1, high: 1)
                if cosThetaI < 0 {
                        refractiveIndex = 1 / refractiveIndex
                        cosThetaI = -cosThetaI
                }
                let sin2ThetaI = 1 - square(cosThetaI)
                let sin2ThetaT = sin2ThetaI / square(refractiveIndex)
                if sin2ThetaT >= 1 {
                        return 1
                }
                let cosThetaT = (1 - sin2ThetaT).squareRoot()
                let refractiveI = refractiveIndex * cosThetaI
                let refractiveT = refractiveIndex * cosThetaT
                let parallel = (refractiveI - cosThetaT) / (refractiveI + cosThetaT)
                let perpendicular = (cosThetaI - refractiveT) / (cosThetaI + refractiveT)
                return (square(parallel) + square(perpendicular)) / 2
        }

        func evaluate(cosTheta: FloatX) -> RGBSpectrum {
                return RGBSpectrum(
                        intensity: evaluate(cosThetaI: cosTheta, refractiveIndex: refractiveIndex))
        }

        var refractiveIndex: FloatX = 1.0
}
