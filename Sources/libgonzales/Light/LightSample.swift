/// When a light is sampled, this is returned.

struct LightSample {
        let radiance: RgbSpectrum
        let direction: Vector
        let pdf: Real
        let visibility: Visibility
}
