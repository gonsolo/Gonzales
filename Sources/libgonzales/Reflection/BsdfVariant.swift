enum BsdfVariant: FramedBsdf {
        case coatedConductorBsdf(CoatedConductorBsdf)
        case coatedDiffuseBsdf(CoatedDiffuseBsdf)
        case dielectricBsdf(DielectricBsdf)
        case diffuseBsdf(DiffuseBsdf)
        case hairBsdf(HairBsdf)
        case microfacetReflection(MicrofacetReflection)
        case mixBsdf(MixBsdf)

        func evaluateWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> RgbSpectrum {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .mixBsdf(let bsdf):
                        return bsdf.evaluateWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                }
        }

        func probabilityDensityWorldSpace(outgoing outgoingWorld: Vector, incident incidentWorld: Vector) -> Real {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .dielectricBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .diffuseBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .hairBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .microfacetReflection(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                case .mixBsdf(let bsdf):
                        return bsdf.probabilityDensityWorldSpace(outgoing: outgoingWorld, incident: incidentWorld)
                }
        }

        func sampleWorldSpace(outgoing outgoingWorld: Vector, uSample: ThreeRandomVariables)
                -> (bsdfSample: BsdfSample, isTransmissive: Bool) {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .dielectricBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .diffuseBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .hairBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .microfacetReflection(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                case .mixBsdf(let bsdf):
                        return bsdf.sampleWorldSpace(outgoing: outgoingWorld, uSample: uSample)
                }
        }

        func evaluate(outgoing: Vector, incident: Vector) -> RgbSpectrum {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .dielectricBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .diffuseBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .hairBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .microfacetReflection(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                case .mixBsdf(let bsdf):
                        return bsdf.evaluate(outgoing: outgoing, incident: incident)
                }
        }

        func albedo() -> RgbSpectrum {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.albedo()
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .dielectricBsdf(let bsdf):
                        return bsdf.albedo()
                case .diffuseBsdf(let bsdf):
                        return bsdf.albedo()
                case .hairBsdf(let bsdf):
                        return bsdf.albedo()
                case .microfacetReflection(let bsdf):
                        return bsdf.albedo()
                case .mixBsdf(let bsdf):
                        return bsdf.albedo()
                }
        }

        var bsdfFrame: BsdfFrame {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .dielectricBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .diffuseBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .hairBsdf(let bsdf):
                        return bsdf.bsdfFrame
                case .microfacetReflection(let bsdf):
                        return bsdf.bsdfFrame
                case .mixBsdf(let bsdf):
                        return bsdf.bsdfFrame
                }
        }

        var isSpecular: Bool {
                switch self {
                case .coatedConductorBsdf(let bsdf):
                        return bsdf.isSpecular
                case .coatedDiffuseBsdf(let bsdf):
                        return bsdf.isSpecular
                case .dielectricBsdf(let bsdf):
                        return bsdf.isSpecular
                case .diffuseBsdf:
                        return false
                case .hairBsdf:
                        return false
                case .microfacetReflection:
                        return false
                case .mixBsdf(let bsdf):
                        return bsdf.isSpecular
                }
        }
}
