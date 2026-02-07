import Foundation  // sin, cos

let sceneDiameter: FloatX = 100.0

struct InfiniteLight {

        init(
                lightToWorld: Transform,
                brightness: RgbSpectrum,
                texture: Texture
        ) {
                self.brightness = brightness
                self.lightToWorld = lightToWorld
                self.texture = texture
        }

        func sample(point: Point, samples: TwoRandomVariables, accelerator _: Accelerator) -> LightSample {
                let theta = samples.1 * FloatX.pi
                let phi = samples.0 * 2 * FloatX.pi
                let lightDirection = Vector(
                        x: sin(theta) * cos(phi),
                        y: sin(theta) * sin(phi),
                        z: cos(theta))
                let direction = lightToWorld * lightDirection
                let pdf = theta < machineEpsilon ? 0 : 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                let distantPoint = point + direction * sceneDiameter
                let visibility = Visibility(from: point, target: distantPoint)
                let uvCoordinates = directionToUV(direction: -direction)
                let interaction = SurfaceInteraction(uvCoordinates: uvCoordinates)
                guard let color = texture.evaluate(at: interaction) as? RgbSpectrum else {
                        print("Unsupported texture type!")
                        return LightSample(
                                radiance: black, direction: direction, pdf: pdf, visibility: visibility)
                }
                return LightSample(radiance: color, direction: direction, pdf: pdf, visibility: visibility)
        }

        func probabilityDensityFor(
                scene _: Scene, samplingDirection direction: Vector, from _: any Interaction
        )
                throws -> FloatX
        {
                let incoming = worldToLight * direction
                let (theta, _) = sphericalCoordinatesFrom(vector: incoming)
                guard theta > machineEpsilon else { return 0 }
                let pdf: FloatX = 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                return pdf
        }

        let inv2Pi: FloatX = 1.0 / (2.0 * FloatX.pi)

        private func sphericalPhi(_ vector: Vector) -> FloatX {
                let phi = atan2(vector.y, vector.x)
                if phi < 0.0 {
                        return phi + 2.0 * FloatX.pi
                } else {
                        return phi
                }
        }

        private func sphericalTheta(_ vector: Vector) -> FloatX {
                return acos(clamp(value: vector.z, low: -1, high: 1))
        }

        private func directionToUV(direction: Vector) -> Point2f {
                let directionLight = normalized(worldToLight * direction)
                let uvCoordinates = equalAreaSphereToSquare(direction: directionLight)
                return uvCoordinates
        }

        private func equalAreaSphereToSquare(direction: Vector) -> Point2f {
                let x = abs(direction.x)
                let y = abs(direction.y)
                let z = abs(direction.z)
                let radius = sqrt(1 - z)
                let coeffA = max(x, y)
                var coeffB = min(x, y)
                if coeffA == 0 {
                        coeffB = 0
                } else {
                        coeffB /= coeffA
                }
                var phi = atan(coeffB) * 2 / FloatX.pi
                if x < y {
                        phi = 1 - phi
                }
                var vCoord: FloatX = phi * radius
                var uCoord: FloatX = radius - vCoord
                if direction.z < 0 {
                        swap(&uCoord, &vCoord)
                        uCoord = 1 - uCoord
                        vCoord = 1 - vCoord
                }
                uCoord = copysign(uCoord, direction.x)
                vCoord = copysign(vCoord, direction.y)
                return Point2f(x: 0.5 * (uCoord + 1), y: 0.5 * (vCoord + 1))
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum {
                let uvCoordinates = directionToUV(direction: ray.direction)
                let interaction = SurfaceInteraction(uvCoordinates: uvCoordinates)
                guard let radiance = texture.evaluate(at: interaction) as? RgbSpectrum else {
                        print("Unsupported texture type!")
                        return black
                }
                return radiance
        }

        func power() -> FloatX {
                let worldRadius = sceneDiameter / 2
                return FloatX.pi * square(worldRadius) * brightness.average()
        }

        var worldToLight: Transform {
                return lightToWorld.inverse
        }

        let brightness: RgbSpectrum
        let lightToWorld: Transform
        let texture: Texture
}

@MainActor
func createInfiniteLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> InfiniteLight
{
        guard let mapname = try parameters.findString(called: "filename") else {
                let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum ?? white
                let constantTexture = ConstantTexture(value: brightness)
                let rgbSpectrumTexture = RgbSpectrumTexture.constantTexture(constantTexture)
                let texture = Texture.rgbSpectrumTexture(rgbSpectrumTexture)
                return InfiniteLight(
                        lightToWorld: lightToWorld,
                        brightness: brightness,
                        texture: texture)
        }
        let texture = try getTextureFrom(name: mapname, type: "color")
        return InfiniteLight(lightToWorld: lightToWorld, brightness: white, texture: texture)
}
