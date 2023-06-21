///        A type that provides photons to the scene and can be sampled.

import Foundation

protocol Light {

        func sample(for reference: Interaction, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        )

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum

        func power() -> FloatX

        var isDelta: Bool { get }
}
