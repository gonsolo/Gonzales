indirect enum GlobalBsdfType: GlobalBsdf {
        case coatedDiffuseBsdf(CoatedDiffuseBsdf)
        case dielectricBsdf(DielectricBsdf)
        case diffuseBsdf(DiffuseBsdf)
        case dummyBsdf(DummyBsdf)
        case geometricPrimitiveBsdf(GlobalBsdfType)  // Special case for recursion
        case hairBsdf(HairBsdf)
        case microfaceReflection(MicrofacetReflection)

        func evaluateWorld(wo woWorld: Vector, wi wiWorld: Vector) -> RgbSpectrum {
                switch self {
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                case .geometricPrimitiveBsdf(let bsdf):
                        return bsdf.evaluateWorld(wo: woWorld, wi: wiWorld)
                default:
                        fatalError("Unhandled GlobalBsdfType case in evaluateWorld")
                }
        }

        func probabilityDensityWorld(wo woWorld: Vector, wi wiWorld: Vector) -> FloatX {
                switch self {
                case .diffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                case .geometricPrimitiveBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(wo: woWorld, wi: wiWorld)
                default:
                        fatalError("Unhandled GlobalBsdfType case in probabilityDensityWorld")
                }
        }

        func sampleWorld(wo woWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
        {
                switch self {

                case .diffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .dielectricBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .hairBsdf(let bsdf):
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                case .geometricPrimitiveBsdf(let bsdf):
                        // Dispatch the call to the wrapped GlobalBsdfType instance
                        return bsdf.sampleWorld(wo: woWorld, u: u)
                default:
                        fatalError("Unhandled GlobalBsdfType case in sampleWorld")
                }
        }

        func evaluateLocal(wo woLocal: Vector, wi wiLocal: Vector) -> RgbSpectrum {
                switch self {
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                case .geometricPrimitiveBsdf(let bsdf):
                        return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                default:
                        fatalError("Unhandled GlobalBsdfType case in evaluateLocal")
                }
        }

        func albedo() -> RgbSpectrum {
                switch self {
                case .diffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .dielectricBsdf(let bsdf):
                        return bsdf.albedo()
                case .geometricPrimitiveBsdf(let bsdf):
                        return bsdf.albedo()
                case .hairBsdf(let bsdf):
                        return bsdf.albedo()
                default:
                        fatalError("Unhandled GlobalBsdfType case in albedo(): \(self)")
                }

                func evaluateLocal(wo woLocal: Vector, wi wiLocal: Vector) -> RgbSpectrum {
                        switch self {
                        case .coatedDiffuseBsdf(let bsdf):
                                return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                        case .dielectricBsdf(let bsdf):
                                return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                        // Add all other concrete BSDF cases...
                        case .geometricPrimitiveBsdf(let bsdf):
                                // Recursion for the special case
                                return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                        case .hairBsdf(let bsdf):
                                return bsdf.evaluateLocal(wo: woLocal, wi: wiLocal)
                        default:
                                fatalError("Unhandled GlobalBsdfType case in evaluateLocal")
                        }
                }
        }

        var bsdfFrame: BsdfFrame {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .dielectricBsdf(let bsdf):
                        return bsdf.bsdfFrame
                // ... add all other concrete BSDF cases ...
                case .geometricPrimitiveBsdf(let bsdf):
                        // Recursion for the special recursive case
                        return bsdf.bsdfFrame
                case .hairBsdf(let bsdf):
                        return bsdf.bsdfFrame
                // Ensure all cases are handled. You must add cases for ConductorBsdf, InterfaceBsdf, etc.
                default:
                        // Use a default frame or trap error if you must guarantee interaction data.
                        fatalError("Unhandled GlobalBsdfType case when accessing bsdfFrame")
                }
        }
}
