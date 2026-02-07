// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations
struct FresnelConductor: Fresnel {

        private func fresnelConductor(
                cosTheta: FloatX,
                etaI: RgbSpectrum,
                etaT: RgbSpectrum,
                k: RgbSpectrum
        ) -> RgbSpectrum {
                let cosThetaClamped = clamp(value: cosTheta, low: -1, high: 1)
                let eta = etaT / etaI
                let etak = k / etaI
                let cosTheta2 = cosThetaClamped * cosThetaClamped
                let sinThetaI2 = 1 - cosTheta2
                let eta2 = eta * eta
                let etak2 = etak * etak
                let term0 = eta2 - etak2 - sinThetaI2
                let a2plusb2 = (term0 * term0 + 4 * eta2 * etak2).squareRoot()
                let term1 = a2plusb2 + cosTheta2
                let a = (0.5 * (a2plusb2 + term0)).squareRoot()
                let term2 = 2 * cosThetaClamped * a
                let reflectanceS = (term1 - term2) / (term1 + term2)
                let term3 = cosTheta2 * a2plusb2 + sinThetaI2 * sinThetaI2
                let term4 = term2 * sinThetaI2
                let reflectanceP = reflectanceS * (term3 - term4) / (term3 + term4)
                return 0.5 * (reflectanceP + reflectanceS)
        }

        func evaluate(cosTheta: FloatX) -> RgbSpectrum {
                return fresnelConductor(cosTheta: abs(cosTheta), etaI: etaI, etaT: etaT, k: k)
        }

        let etaI: RgbSpectrum
        let etaT: RgbSpectrum
        let k: RgbSpectrum
}
