import Foundation

public struct CoatedDiffuseBsdf: GlobalBsdf {

        private let layered: LayeredBsdf<DielectricBsdf, DiffuseBsdf>
        private let _albedo: RgbSpectrum
        public let bsdfFrame: BsdfFrame

        public init(
                reflectance: RgbSpectrum,
                refractiveIndex: FloatX,
                roughness: (FloatX, FloatX),
                remapRoughness: Bool,
                bsdfFrame: BsdfFrame
        ) {
                self._albedo = reflectance
                self.bsdfFrame = bsdfFrame

                let alpha = remapRoughness ? TrowbridgeReitzDistribution.getAlpha(from: roughness) : roughness
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let topBxdf = DielectricBsdf(
                        distribution: distribution,
                        refractiveIndex: refractiveIndex,
                        bsdfFrame: bsdfFrame
                )

                let bottomBxdf = DiffuseBsdf(
                        reflectance: reflectance,
                        bsdfFrame: bsdfFrame
                )

                let thickness: FloatX = 0.01
                let g: FloatX = 0.0
                let maxDepth = 10
                let nSamples = 1

                let mediumAlbedo = RgbSpectrum(intensity: 0.0)

                self.layered = LayeredBsdf(
                        top: topBxdf,
                        bottom: bottomBxdf,
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
                return _albedo
        }

        var isSpecular: Bool {
                return true
        }
}
