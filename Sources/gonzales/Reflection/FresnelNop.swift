struct FresnelNop: Fresnel {

        func evaluate(cosTheta: FloatX) -> RGBSpectrum { return white }
}
