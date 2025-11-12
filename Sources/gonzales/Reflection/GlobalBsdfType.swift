enum GlobalBsdfType: GlobalBsdf {
        case coatedDiffuseBsdf(CoatedDiffuseBsdf)
        case dielectricBsdf(DielectricBsdf)
        case diffuseBsdf(DiffuseBsdf)
        case dummyBsdf(DummyBsdf)
        case hairBsdf(HairBsdf)
        case microfacetReflection(MicrofacetReflection)

        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RgbSpectrum {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .dummyBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                }
        }

        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .dummyBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                }
        }

        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool) {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .dielectricBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .diffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .dummyBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .hairBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .microfacetReflection(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                }
        }

        func evaluateLocal(wo woLocal: Vector, wi wiLocal: Vector) -> RgbSpectrum {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .dummyBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                }
        }

        func albedo() -> RgbSpectrum {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .dielectricBsdf(let bsdf):
                        return bsdf.albedo()
                case .diffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .dummyBsdf(let bsdf):
                        return bsdf.albedo()
                case .hairBsdf(let bsdf):
                        return bsdf.albedo()
                case .microfacetReflection(let bsdf):
                        return bsdf.albedo()
                }
        }

        var bsdfFrame: BsdfFrame {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .dielectricBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .diffuseBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .dummyBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .hairBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .microfacetReflection(let bsdf):
                        return bsdf.bsdfFrame
                }
        }
}
