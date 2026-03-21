import Foundation

public struct CoatedDiffuseBsdf: FramedBsdf {

        private let layered: LayeredBsdf<DielectricBsdf, DiffuseBsdf>
        public let bsdfFrame: BsdfFrame

        public init(
                dielectric: DielectricBsdf,
                diffuse: DiffuseBsdf,
                thickness: Real,
                albedo: RgbSpectrum,
                g: Real,
                maxDepth: Int,
                nSamples: Int,
                bsdfFrame: BsdfFrame
        ) {
                self.layered = LayeredBsdf(
                        top: dielectric,
                        bottom: diffuse,
                        topIsSpecular: dielectric.isSpecular,
                        bottomIsSpecular: diffuse.isSpecular,
                        thickness: thickness,
                        albedo: albedo,
                        geometricTerm: g,
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        bsdfFrame: bsdfFrame,
                        twoSided: true
                )
                self.bsdfFrame = bsdfFrame
        }

        public func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                return layered.evaluate(outgoing: outgoing, incident: incident)
        }

        public func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                return layered.sample(outgoing: outgoing, uSample: uSample)
        }

        public func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                return layered.probabilityDensity(outgoing: outgoing, incident: incident)
        }

        public func albedo() -> RgbSpectrum {
                return layered.bottom.albedo()
        }

        var isSpecular: Bool {
                return false
        }
}
