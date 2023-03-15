protocol Medium {

        // Sample the medium along the ray. If the proposed sample is behind the
        // max length just return transmittance and nil, otherwise the medium is
        // sampled and transmittance as well as an interaction is returned.
        func sample(ray: Ray, tHit: FloatX, sampler: Sampler) -> (RGBSpectrum, MediumInteraction?)
}
