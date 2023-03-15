import Foundation

final class Homogeneous: Medium {

        init(scale: FloatX, absorption: RGBSpectrum, scattering: RGBSpectrum) {
                self.scale = scale
                self.absorption = absorption
                self.scattering = scattering
        }

        func sample(ray: Ray, tHit: FloatX, sampler: Sampler) -> (RGBSpectrum, MediumInteraction?) {
                let channel = Int(sampler.get1D() * 3)
                let transmission = absorption + scattering
                let transmissionChannel = transmission[channel]
                let distance = -log(1 - sampler.get1D()) / transmissionChannel
                let rayParameter = min(distance / length(ray.direction), tHit)
                let sampledMedium = rayParameter < tHit
                let interaction: MediumInteraction? = sampledMedium ? MediumInteraction() : nil
                let transmittance = exp(-transmission * rayParameter * length(ray.direction))
                let density = sampledMedium ? transmission * transmittance : transmittance
                let probabilityDensity = density.average()
                let transmittancePerDensity =
                        sampledMedium
                        ? transmittance * transmission / probabilityDensity
                        : transmittance / probabilityDensity
                return (transmittancePerDensity, interaction)
        }

        let scale: FloatX
        let absorption: RGBSpectrum
        let scattering: RGBSpectrum
}
