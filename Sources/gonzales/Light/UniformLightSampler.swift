struct UniformLightSampler: Sendable {

        init(sampler: RandomSampler, lights: [Light]) {
                self.sampler = sampler
                self.lights = lights
        }

        func chooseLight() -> (Light, FloatX) {
                assert(lights.count > 0)
                let u = sampler.get1D()
                let lightNum = Int(u * FloatX(lights.count))
                let light = lights[lightNum]
                let probabilityDensity: FloatX = 1.0 / FloatX(lights.count)
                return (light, probabilityDensity)
        }

        let sampler: RandomSampler
        let lights: [Light]
}
