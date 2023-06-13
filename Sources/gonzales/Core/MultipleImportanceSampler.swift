struct MultipleImportanceSampler {

        struct MISSampler {
                typealias sampleFunc = (Light, Interaction, Sampler)
                        throws -> BsdfSample

                typealias densityFunc = (Light, Interaction, Vector) throws -> FloatX

                let sample: sampleFunc
                let density: densityFunc
        }

        func evaluate(
                light: Light,
                interaction: Interaction,
                sampler: Sampler
        ) throws -> RGBSpectrum {

                func evaluate(
                        first: MISSampler,
                        second: MISSampler,
                        light: Light,
                        interaction: Interaction,
                        sampler: Sampler
                ) throws -> RGBSpectrum {
                        let thisSample = try first.sample(
                                light, interaction, sampler)
                        let otherDensity = try second.density(
                                light, interaction, thisSample.incoming)
                        let weight = powerHeuristic(f: thisSample.probabilityDensity, g: otherDensity)
                        return thisSample.probabilityDensity == 0
                                ? black : thisSample.estimate * weight / thisSample.probabilityDensity
                }

                let a = try evaluate(
                        first: samplers.0,
                        second: samplers.1,
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                let b = try evaluate(
                        first: samplers.1,
                        second: samplers.0,
                        light: light,
                        interaction: interaction,
                        sampler: sampler)
                return a + b
        }

        let scene: Scene
        let samplers: (MISSampler, MISSampler)
}
