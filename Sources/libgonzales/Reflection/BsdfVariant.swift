enum BsdfVariant: FramedBsdf {
        case coatedConductorBsdf(CoatedConductorBsdf)
        case coatedDiffuseBsdf(CoatedDiffuseBsdf)
        case dielectricBsdf(DielectricBsdf)
        case diffuseBsdf(DiffuseBsdf)
        case hairBsdf(HairBsdf)
        case microfacetReflection(MicrofacetReflection)
        case mixBsdf(MixBsdf)
}
