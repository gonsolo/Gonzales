func frDielectric(cosThetaI: FloatX, etaI: FloatX, etaT: FloatX) -> FloatX {
        var etaI = etaI
        var etaT = etaT
        var cosThetaI = clamp(value: cosThetaI, low: -1, high: 1)
        let entering = cosThetaI > 0
        if !entering {
                swap(&etaI, &etaT)
                cosThetaI = abs(cosThetaI)
        }
        let sinThetaI = (max(0, 1 - cosThetaI * cosThetaI)).squareRoot()
        let sinThetaT = etaI / etaT * sinThetaI
        if sinThetaI >= 1 { return 1 }
        let cosThetaT = (max(0, 1 - sinThetaT * sinThetaT)).squareRoot()
        let ti = etaT * cosThetaI
        let ii = etaI * cosThetaI
        let tt = etaT * cosThetaT
        let it = etaI * cosThetaT
        let parallel = (ti - it) / (ti + it)
        let perpendicular = (ii - tt) / (ii + tt)
        return (parallel * parallel + perpendicular * perpendicular) / 2
}

struct FresnelDielectric: Fresnel {

        func evaluate(cosTheta: FloatX) -> Spectrum {
                return Spectrum(
                        intensity: frDielectric(cosThetaI: cosTheta, etaI: etaI, etaT: etaT))
        }

        var etaI: FloatX = 1.0
        var etaT: FloatX = 1.0
}
