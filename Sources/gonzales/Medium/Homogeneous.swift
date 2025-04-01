import Foundation

final class Homogeneous: Medium {

        init(scale: FloatX, absorption: RgbSpectrum, scattering: RgbSpectrum) {
                self.scale = scale
                self.absorption = absorption
                self.scattering = scattering
        }

        func sample(ray: Ray, tHit: FloatX, sampler: any Sampler) -> (RgbSpectrum, MediumInteraction?) {
                let channel = Int(sampler.get1D() * 3)
                let transmission = absorption + scattering
                let transmissionChannel = transmission[channel]
                let distance = -log(1 - sampler.get1D()) / transmissionChannel
                let rayParameter = min(distance / length(ray.direction), tHit)
                let sampledMedium = rayParameter < tHit
                let position = ray.origin + rayParameter * ray.direction
                let interaction: MediumInteraction? =
                        sampledMedium ? MediumInteraction(position: position, wo: -ray.direction) : nil
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
        let absorption: RgbSpectrum
        let scattering: RgbSpectrum
}
