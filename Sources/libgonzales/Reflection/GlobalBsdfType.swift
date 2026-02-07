enum GlobalBsdfType: GlobalBsdf {
        case coatedDiffuseBsdf(CoatedDiffuseBsdf)
        case dielectricBsdf(DielectricBsdf)
        case diffuseBsdf(DiffuseBsdf)
        case dummyBsdf(DummyBsdf)
        case hairBsdf(HairBsdf)
        case microfacetReflection(MicrofacetReflection)

        func evaluateWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .dummyBsdf(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluateWorld(outgoing: outgoingWorld, incident: incidentWorld)
                }
        }

        func probabilityDensityWorld(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> FloatX {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .dummyBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.probabilityDensityWorld(outgoing: outgoingWorld, incident: incidentWorld)
                }
        }

        func sampleWorld(outgoing outgoingWorld: Vector, u: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool)
        {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                case .dielectricBsdf(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                case .diffuseBsdf(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                case .dummyBsdf(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                case .hairBsdf(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                case .microfacetReflection(let bsdf):
                        return bsdf.sampleWorld(outgoing: outgoingWorld, u: u)
                }
        }

        func evaluateLocal(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                switch self {
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
                case .dummyBsdf(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluateLocal(outgoing: outgoing, incident: incident)
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
