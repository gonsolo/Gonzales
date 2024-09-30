final class UniformLightSampler: Sendable {

        init(sampler: Sampler, lights: [Light]) {
                self.sampler = sampler
                self.lights = lights
        }

        func chooseLight() async -> (Light, FloatX) {
                assert(lights.count > 0)
                let u = await sampler.get1D()
                let lightNum = Int(u * FloatX(lights.count))
                let light = lights[lightNum]
                let probabilityDensity: FloatX = 1.0 / FloatX(lights.count)
                return (light, probabilityDensity)
        }

        let sampler: Sampler
        let lights: [Light]
}
