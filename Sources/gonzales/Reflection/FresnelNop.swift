struct FresnelNop: Fresnel {

        func evaluate(cosTheta: FloatX) -> Spectrum { return white }
}
