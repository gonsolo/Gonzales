final class Homogeneous: Medium {

        init(scale: FloatX, absorption: RGBSpectrum, scattering: RGBSpectrum) {
                self.scale = scale
                self.absorption = absorption
                self.scattering = scattering
        }

        let scale: FloatX
        let absorption: RGBSpectrum
        let scattering: RGBSpectrum
}
