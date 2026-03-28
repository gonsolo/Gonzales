import Foundation  // sin, cos

let sceneDiameter: Real = 100.0

struct InfiniteLight: LightSource {

        init(
                lightToWorld: Transform,
                brightness: RgbSpectrum,
                texture: Texture,
                arena: TextureArena
        ) {
                self.brightness = brightness
                self.lightToWorld = lightToWorld
                self.texture = texture

                let width = 1024
                let height = 1024
                var data = [Real]()
                data.reserveCapacity(width * height)

                for row in 0..<height {
                        for col in 0..<width {
                                let uvCoordinates = Point2f(
                                        x: (Real(col) + 0.5) / Real(width),
                                        y: (Real(row) + 0.5) / Real(height))
                                let interaction = SurfaceInteraction(uvCoordinates: uvCoordinates)
                                let color = texture.evaluateRgbSpectrum(at: interaction, arena: arena)
                                if true {
                                        data.append(color.y)
                                } else {
                                        data.append(0)
                                }
                        }
                }
                self.distribution = PiecewiseConstant2D(data: data, width: width, height: height)
        }

        func sample(
                point: Point, samples: TwoRandomVariables,
                accelerator _: Accelerator, scene: Scene
        ) -> LightSample {
                let (texCoord, mapPdf) = distribution.sampleContinuous(
                        sample: Point2f(x: samples.0, y: samples.1))
                if mapPdf == 0 {
                        return LightSample(
                                radiance: black, direction: Vector(), pdf: 0,
                                visibility: Visibility(from: point, target: point))
                }

                let lightDirection = equalAreaSquareToSphere(texCoord: texCoord)
                let direction = lightToWorld * lightDirection

                let pdf = mapPdf / (4 * Real.pi)
                let distantPoint = point + direction * sceneDiameter
                let visibility = Visibility(from: point, target: distantPoint)

                let interaction = SurfaceInteraction(uvCoordinates: texCoord)
                guard let color = texture.evaluate(at: interaction, arena: scene.arena) as? RgbSpectrum else {
                        return LightSample(
                                radiance: black, direction: direction, pdf: pdf, visibility: visibility)
                }

                return LightSample(radiance: color, direction: direction, pdf: pdf, visibility: visibility)
        }

        func probabilityDensityFor<I: Interaction>(
                scene _: Scene, samplingDirection direction: Vector, from _: I
        )
                -> Real
        {
                let incoming = worldToLight * direction
                let texCoord = equalAreaSphereToSquare(direction: incoming)
                let mapPdf = distribution.pdf(texCoord: texCoord)
                return mapPdf / (4 * Real.pi)
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
                let radius = sqrt(max(0, 1 - z))
                let coeffA = max(x, y)
                var coeffB = min(x, y)
                if coeffA == 0 {
                        coeffB = 0
                } else {
                        coeffB /= coeffA
                }
                var phi = atan(coeffB) * 2 / Real.pi
                if x < y {
                        phi = 1 - phi
                }
                var vCoord: Real = phi * radius
                var uCoord: Real = radius - vCoord
                if direction.z < 0 {
                        swap(&uCoord, &vCoord)
                        uCoord = 1 - uCoord
                        vCoord = 1 - vCoord
                }
                uCoord = copysign(uCoord, direction.x)
                vCoord = copysign(vCoord, direction.y)
                return Point2f(x: 0.5 * (uCoord + 1), y: 0.5 * (vCoord + 1))
        }

        private func equalAreaSquareToSphere(texCoord: Point2f) -> Vector {
                let remappedU = 2 * texCoord.x - 1
                let remappedV = 2 * texCoord.y - 1
                let absU = abs(remappedU)
                let absV = abs(remappedV)
                let signedDistance = 1 - (absU + absV)
                let absDist = abs(signedDistance)
                let radius = 1 - absDist
                let phi = (radius == 0 ? 1 : (absV - absU) / radius + 1) * (Real.pi / 4)
                let zCoord = copysign(max(0, 1 - square(radius)), signedDistance)
                let cosPhi = cos(phi)
                let sinPhi = sin(phi)
                let xCoord = copysign(radius * cosPhi, remappedU)
                let yCoord = copysign(radius * sinPhi, remappedV)
                return Vector(x: xCoord, y: yCoord, z: zCoord)
        }

        func radianceFromInfinity(for ray: Ray, arena: TextureArena) -> RgbSpectrum {
                let uvCoordinates = directionToUV(direction: ray.direction)
                let interaction = SurfaceInteraction(uvCoordinates: uvCoordinates)
                return texture.evaluateRgbSpectrum(at: interaction, arena: arena)
        }

        func power(scene _: Scene) -> Real {
                let worldRadius = sceneDiameter / 2
                return Real.pi * square(worldRadius) * brightness.average()
        }

        var worldToLight: Transform {
                return lightToWorld.inverse
        }

        let brightness: RgbSpectrum
        let lightToWorld: Transform
        let texture: Texture
        let distribution: PiecewiseConstant2D
}

extension InfiniteLight {
        static func create(
                lightToWorld: Transform, parameters: ParameterDictionary, sceneDirectory: String,
                arena: inout TextureArena
        ) throws
                -> InfiniteLight
        {
                guard let mapname = try parameters.findString(called: "filename") else {
                        let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum ?? white
                        let constantTexture = ConstantTexture(value: brightness)
                        let rgbSpectrumTexture = RgbSpectrumTexture.constantTexture(constantTexture)
                        let index = arena.appendRgb(rgbSpectrumTexture)
                        let texture = Texture.rgbSpectrumTexture(index)
                        return InfiniteLight(
                                lightToWorld: lightToWorld,
                                brightness: brightness,
                                texture: texture,
                                arena: arena)
                }
                let scale = try parameters.findOneReal(called: "scale", else: 1)
                let texture = try getTextureFrom(
                        name: mapname, type: "color", sceneDirectory: sceneDirectory, arena: &arena)
                return InfiniteLight(
                        lightToWorld: lightToWorld, brightness: RgbSpectrum(intensity: scale),
                        texture: texture, arena: arena)
        }
}

extension InfiniteLight {
        var isDelta: Bool { false }
}
