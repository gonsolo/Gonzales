import Foundation

struct DistantLight: LightSource {

        init(lightToWorld: Transform, brightness: RgbSpectrum, direction: Vector) {
                self.brightness = brightness
                self.direction = normalized(lightToWorld * direction)
        }

        func sample(point: Point, samples _: TwoRandomVariables, accelerator _: Accelerator, scene _: Scene)
                -> LightSample
        {  // radiance: RgbSpectrum, direction: Vector, pdf: Real, visibility: Visibility
                let outside = point + direction * 2 * worldRadius
                let visibility = Visibility(
                        from: point, target: outside)
                return LightSample(radiance: brightness, direction: direction, pdf: 1, visibility: visibility)
        }

        func probabilityDensityFor<I: Interaction>(
                scene _: Scene, samplingDirection _: Vector, from _: I
        )
                -> Real
        {
                return 0
        }

        func radianceFromInfinity(for _: Ray, arena _: TextureArena) -> RgbSpectrum { return black }

        func power(scene _: Scene) -> Real {
                let value = square(worldRadius) * Real.pi * brightness.average()
                return value
        }

        let direction: Vector
        let brightness: RgbSpectrum
        let worldRadius: Real = 100.0
}

extension DistantLight {
        static func create(lightToWorld: Transform, parameters: ParameterDictionary) throws
                -> DistantLight
        {
                let from = try parameters.findOnePoint(name: "from", else: origin)
                let target = try parameters.findOnePoint(name: "to", else: origin)
                guard let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum else {
                        throw ParameterError.missing(parameter: "L", function: #function)
                }
                let direction: Vector = from - target
                return DistantLight(
                        lightToWorld: lightToWorld, brightness: brightness, direction: direction)
        }
}

extension DistantLight {
        var isDelta: Bool { true }
}
