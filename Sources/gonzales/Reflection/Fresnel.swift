///        A type that computes the amount of light that is reflected at from a
///        specific direction.
protocol Fresnel {
        func evaluate(cosTheta: FloatX) -> RGBSpectrum
}
