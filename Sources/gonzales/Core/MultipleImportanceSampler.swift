struct MultipleImportanceSampler<Sample> {

        struct MISSampler {
                typealias sampleFunc = (Light, Interaction, Sampler, BSDF, Scene) throws -> (
                        estimate: Spectrum,
                        density: FloatX,
                        sample: Sample
                )

                typealias densityFunc = (Light, Interaction, Sample, BSDF) throws -> FloatX

                let sample: sampleFunc
                let density: densityFunc
        }

        func evaluate() throws -> Spectrum {

                func evaluate(first: MISSampler, second: MISSampler) throws -> Spectrum {
                        let (estimate, density, sample) = try first.sample(
                                light, interaction, sampler, bsdf, scene)
                        let otherDensity = try second.density(
                                light, interaction, sample, bsdf)
                        let weight = powerHeuristic(f: density, g: otherDensity)
                        return density == 0 ? black : estimate * weight / density
                }

                let a = try evaluate(first: samplers.0, second: samplers.1)
                let b = try evaluate(first: samplers.1, second: samplers.0)
                return a + b
        }

        let samplers: (MISSampler, MISSampler)
        let light: Light
        let interaction: Interaction
        let sampler: Sampler
        let bsdf: BSDF
        unowned let scene: Scene
}
