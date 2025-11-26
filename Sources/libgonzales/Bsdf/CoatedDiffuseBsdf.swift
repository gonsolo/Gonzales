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
                let nSamples = 1
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

        public func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum {
                let value = layered.evaluateLocal(wo: wo, wi: wi)
                return value
        }

        func sampleLocal(wo: Vector, u: ThreeRandomVariables) async -> BsdfSample {
                return await layered.sampleLocal(wo: wo, u: u)
        }

        public func probabilityDensityLocal(wo: Vector, wi: Vector) async -> FloatX {
                return await layered.probabilityDensityLocal(wo: wo, wi: wi)
        }

        public func albedo() -> RgbSpectrum {
                return layered.bottom.albedo()
        }

        var isSpecular: Bool {
                return true
        }
}
