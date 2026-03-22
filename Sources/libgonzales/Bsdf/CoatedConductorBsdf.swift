import Foundation

struct CoatedConductorBsdf: FramedBsdf {

        private let layered: LayeredBsdf<DielectricBsdf, MicrofacetReflection>
        let bsdfFrame: BsdfFrame

        init(
                dielectric: DielectricBsdf,
                conductor: MicrofacetReflection,
                thickness: Real,
                albedo: RgbSpectrum,
                asymmetry: Real,
                maxDepth: Int,
                nSamples: Int,
                bsdfFrame: BsdfFrame
        ) {
                // Dielectric (clearcoat) is completely specular (smooth or microfacet)
                // Conductor is treated equivalently underneath.
                self.layered = LayeredBsdf(
                        top: dielectric,
                        bottom: conductor,
                        topIsSpecular: dielectric.isSpecular,
                        bottomIsSpecular: false,
                        thickness: thickness,
                        albedo: albedo,
                        asymmetry: asymmetry,
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        bsdfFrame: bsdfFrame,
                        twoSided: true
                )
                self.bsdfFrame = bsdfFrame
        }

        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                return layered.evaluate(outgoing: outgoing, incident: incident)
        }

        func sample(outgoing: Vector, uSample: ThreeRandomVariables) -> BsdfSample {
                return layered.sample(outgoing: outgoing, uSample: uSample)
        }

        func probabilityDensity(outgoing: Vector, incident: Vector) -> Real {
                return layered.probabilityDensity(outgoing: outgoing, incident: incident)
        }

        func albedo() -> RgbSpectrum {
                return layered.bottom.albedo()
        }

        var isSpecular: Bool {
                // In layer structures, a purely smooth clearcoat guarantees overall specularity if bottom is missing
                return false
        }
}
