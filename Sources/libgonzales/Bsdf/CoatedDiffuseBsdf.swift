import Foundation

public struct CoatedDiffuseBsdf: GlobalBsdf {

        private let layered: LayeredBsdf<DielectricBsdf, DiffuseBsdf>
        public let bsdfFrame: BsdfFrame

        public init(
                dielectric: DielectricBsdf,
                diffuse: DiffuseBsdf,
                bsdfFrame: BsdfFrame
        ) {
                let thickness: FloatX = 0.01
                let g: FloatX = 0.0
                let maxDepth = 10
                let nSamples = 32
                let mediumAlbedo = RgbSpectrum(intensity: 0.0)

                self.layered = LayeredBsdf(
                        top: dielectric,
                        bottom: diffuse,
                        topIsSpecular: true,
                        bottomIsSpecular: false,
                        thickness: thickness,
                        albedo: mediumAlbedo,
                        g: g,
                        maxDepth: maxDepth,
                        nSamples: nSamples,
                        bsdfFrame: bsdfFrame,
                        twoSided: true
                )
                self.bsdfFrame = bsdfFrame
        }

        public func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                let value = layered.evaluateLocal(outgoing: outgoing, incident: incident)
                return value
        }

        func sampleLocal(outgoing: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                return await layered.sampleLocal(outgoing: outgoing, u: u)
        }

        public func probabilityDensityLocal(outgoing: Vector, incident: Vector) async -> FloatX {
                return await layered.probabilityDensityLocal(outgoing: outgoing, incident: incident)
        }

        public func albedo() -> RgbSpectrum {
                return layered.bottom.albedo()
        }

        var isSpecular: Bool {
                return true
        }
}
