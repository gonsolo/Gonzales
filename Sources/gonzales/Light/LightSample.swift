/// When a light is sampled, this is returned.

struct LightSample {
    let radiance: RgbSpectrum
    let direction: Vector
    let pdf: FloatX
    let visibility: Visibility
}
