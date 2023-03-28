///        A type that provides photons to the scene and can be sampled.

import Foundation

//typealias Power = FloatX

protocol Light {

        func sample(for reference: Interaction, u: TwoRandomVariables) -> (
                radiance: RGBSpectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        )

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX

        func radianceFromInfinity(for ray: Ray) -> RGBSpectrum

        func power() -> Measurement<UnitPower>

        var isDelta: Bool { get }
}
