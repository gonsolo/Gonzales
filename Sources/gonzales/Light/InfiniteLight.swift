import Foundation  // sin, cos

// Is initialized in Scene.init.
// TODO: Move somewhere reasonable.
var sceneDiameter: FloatX = 100.0

struct InfiniteLight: Light {

        init(
                lightToWorld: Transform,
                brightness: RgbSpectrum,
                texture: Texture
        ) {
                self.lightToWorld = lightToWorld
                self.brightness = brightness
                self.texture = texture
        }

        func sample(for reference: Interaction, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum,
                direction: Vector,
                pdf: FloatX,
                visibility: Visibility
        ) {
                let theta = u.1 * FloatX.pi
                let phi = u.0 * 2 * FloatX.pi
                let lightDirection = Vector(
                        x: sin(theta) * cos(phi),
                        y: sin(theta) * sin(phi),
                        z: cos(theta))
                let direction = lightToWorld * lightDirection
                let pdf = theta < machineEpsilon ? 0 : 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                let distantPoint = SurfaceInteraction(
                        position: reference.position + direction * sceneDiameter)
                let visibility = Visibility(from: reference, to: distantPoint)
                let uv = directionToUV(direction: -direction)
                let interaction = SurfaceInteraction(uv: uv)
                guard let color = texture.evaluate(at: interaction) as? RgbSpectrum else {
                        warning("Unsupported texture type!")
                        return (black, direction, pdf, visibility)
                }
                return (radiance: color, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                let incoming = worldToLight * direction
                let (theta, _) = sphericalCoordinatesFrom(vector: incoming)
                guard theta > machineEpsilon else { return 0 }
                let pdf: FloatX = 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                return pdf
        }

        let inv2Pi: FloatX = 1.0 / (2.0 * FloatX.pi)

        private func sphericalPhi(_ v: Vector) -> FloatX {
                let p = atan2(v.y, v.x)
                if p < 0.0 {
                        return p + 2.0 * FloatX.pi
                } else {
                        return p
                }
        }

        private func sphericalTheta(_ v: Vector) -> FloatX {
                return acos(clamp(value: v.z, low: -1, high: 1))
        }

        private func directionToUV(direction: Vector) -> Point2F {
                let directionLight = normalized(worldToLight * direction)
                let uv = equalAreaSphereToSquare(direction: directionLight)
                return uv
        }

        private func equalAreaSphereToSquare(direction: Vector) -> Point2F {
                let x = abs(direction.x)
                let y = abs(direction.y)
                let z = abs(direction.z)
                let r = sqrt(1 - z)
                let a = max(x, y)
                var b = min(x, y)
                if a == 0 {
                        b = 0
                } else {
                        b = b / a
                }
                var phi = atan(b) * 2 / FloatX.pi
                if x < y {
                        phi = 1 - phi
                }
                var v: FloatX = phi * r
                var u: FloatX = r - v
                if direction.z < 0 {
                        swap(&u, &v)
                        u = 1 - u
                        v = 1 - v
                }
                u = copysign(u, direction.x)
                v = copysign(v, direction.y)
                return Point2F(x: 0.5 * (u + 1), y: 0.5 * (v + 1))
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum {
                let uv = directionToUV(direction: ray.direction)
                let interaction = SurfaceInteraction(uv: uv)
                guard let radiance = texture.evaluate(at: interaction) as? RgbSpectrum else {
                        warning("Unsupported texture type!")
                        return black
                }
                return radiance
        }

        func power() -> FloatX {
                let worldRadius = sceneDiameter / 2
                return FloatX.pi * square(worldRadius) * brightness.average()
        }

        var isDelta: Bool { return false }

        var worldToLight: Transform { return lightToWorld.inverse }

        let brightness: RgbSpectrum
        let texture: Texture
        let lightToWorld: Transform
}

func createInfiniteLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> InfiniteLight
{
        guard let mapname = try parameters.findString(called: "filename") else {
                let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum ?? white
                let texture = ConstantTexture(value: brightness)
                return InfiniteLight(
                        lightToWorld: lightToWorld,
                        brightness: brightness,
                        texture: texture)
        }
        let texture = try getTextureFrom(name: mapname, type: "color")
        return InfiniteLight(lightToWorld: lightToWorld, brightness: white, texture: texture)
}
