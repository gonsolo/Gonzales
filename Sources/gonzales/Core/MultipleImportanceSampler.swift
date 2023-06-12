struct MultipleImportanceSampler {

        struct MISSampler {
                typealias sampleFunc = (Light, Interaction, Sampler, GlobalBsdf)
                        throws -> BsdfSample

                typealias densityFunc = (Light, Interaction, Vector, GlobalBsdf) throws -> FloatX

                let sample: sampleFunc
                let density: densityFunc
        }

        func evaluate(
                light: Light,
                interaction: Interaction,
                sampler: Sampler,
                bsdf: GlobalBsdf
        ) throws -> RGBSpectrum {

                func evaluate(
                        first: MISSampler,
                        second: MISSampler,
                        light: Light,
                        interaction: Interaction,
                        sampler: Sampler,
                        bsdf: GlobalBsdf
                ) throws -> RGBSpectrum {
                        let thisSample = try first.sample(
                                light, interaction, sampler, bsdf)
                        let otherDensity = try second.density(
                                light, interaction, thisSample.incoming, bsdf)
                        let weight = powerHeuristic(f: thisSample.probabilityDensity, g: otherDensity)
                        return thisSample.probabilityDensity == 0
                                ? black : thisSample.estimate * weight / thisSample.probabilityDensity
                }

                let a = try evaluate(
                        first: samplers.0,
                        second: samplers.1,
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
                let b = try evaluate(
                        first: samplers.1,
                        second: samplers.0,
                        light: light,
                        interaction: interaction,
                        sampler: sampler,
                        bsdf: bsdf)
                return a + b
        }

        let scene: Scene
        let samplers: (MISSampler, MISSampler)
}
