final class Homogeneous: Medium {

        init(scale: FloatX, absorption: RGBSpectrum, scattering: RGBSpectrum) {
                self.scale = scale
                self.absorption = absorption
                self.scattering = scattering
        }

        func sample() -> (RGBSpectrum, MediumInteraction) {
                //unimplemented()
                return (white, MediumInteraction())
        }

        let scale: FloatX
        let absorption: RGBSpectrum
        let scattering: RGBSpectrum
}
