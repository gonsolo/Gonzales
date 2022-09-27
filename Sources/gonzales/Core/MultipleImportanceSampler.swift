struct MultipleImportanceSampler<Sample> {

        struct Sampler {
                typealias sampleFunc = () throws -> (
                        estimate: Spectrum,
                        density: FloatX,
                        sample: Sample
                )

                typealias densityFunc = (_ sample: Sample) throws -> FloatX

                let sample: sampleFunc
                let density: densityFunc
        }

        func evaluate() throws -> Spectrum {

                func evaluate(sampler: Sampler, other: Sampler) throws -> Spectrum {
                        let (estimate, density, sample) = try sampler.sample()
                        let otherDensity = try other.density(sample)
                        let weight = powerHeuristic(f: density, g: otherDensity)
                        return density == 0 ? black : estimate * weight / density
                }

                let a = try evaluate(sampler: samplers.0, other: samplers.1)
                let b = try evaluate(sampler: samplers.1, other: samplers.0)
                return a + b
        }

        let samplers: (Sampler, Sampler)
}
