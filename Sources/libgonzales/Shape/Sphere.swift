import Foundation

struct Sphere: Shape {

        init(radius: Real, objectToWorld: Transform) {
                self.radius = radius
                self.objectToWorld = objectToWorld
        }

        var description: String {
                return "Sphere"
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        func objectBound(scene _: Scene) -> Bounds3f {

                let point0 = Point(x: -radius, y: -radius, z: -radius)
                let point1 = Point(x: +radius, y: +radius, z: +radius)

                return Bounds3f(first: point0, second: point1)
        }

        func uniformSampleSphere(samples: TwoRandomVariables) -> Point {

                let zComponent = 1 - 2 * samples.0
                let radiusValue = max(0, 1 - zComponent * zComponent).squareRoot()
                let phi = 2 * Real.pi * samples.1
                return Point(x: radiusValue * cos(phi), y: radiusValue * sin(phi), z: zComponent)
        }

        func sample(samples: TwoRandomVariables, scene _: Scene) -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {

                let localPosition = radius * uniformSampleSphere(samples: samples)
                let worldNormal = normalized(objectToWorld * Normal(point: localPosition))
                let worldPosition = objectToWorld * localPosition
                let interaction = SurfaceInteraction(position: worldPosition, normal: worldNormal)
                let pdf: Real = 1.0
                return (interaction, pdf)
        }

        func sample(ref: any Interaction, samples: TwoRandomVariables, scene: Scene)
                -> (interaction: any Interaction, pdf: Real) {
                let center = objectToWorld * origin
                if distanceSquared(ref.position, center) <= radius * radius {
                        var (interaction, pdf) = sample(samples: samples, scene: scene)
                        var incident: Vector = interaction.position - ref.position
                        if lengthSquared(incident) == 0 {
                                pdf = 0
                        } else {
                                incident = normalized(incident)
                                pdf *=
                                        distanceSquared(ref.position, interaction.position)
                                        / absDot(interaction.normal, Normal(-incident))
                        }
                        if pdf.isInfinite { pdf = 0 }
                        return (interaction, pdf)
                }
                let distanceCenter = distance(ref.position, center)
                let invDc = 1 / distanceCenter
                let vectorCenter = Vector(vector: center - ref.position) * invDc
                let (wcX, wcY) = makeCoordinateSystem(from: vectorCenter)
                let sinThetaMax = radius * invDc
                let sinThetaMax2 = sinThetaMax * sinThetaMax
                let invSinThetaMax = 1 / sinThetaMax
                let cosThetaMax = max(0, 1 - sinThetaMax2).squareRoot()
                var cosTheta = (cosThetaMax - 1) * samples.0 + 1
                var sinTheta2 = 1 - cosTheta * cosTheta
                if sinThetaMax2 < 0.00068523 {
                        sinTheta2 = sinThetaMax2 * samples.0
                        cosTheta = (1 - sinTheta2).squareRoot()
                }
                let cosAlpha =
                        sinTheta2 * invSinThetaMax + cosTheta
                        * (max(0, 1 - sinTheta2 * invSinThetaMax * invSinThetaMax)).squareRoot()
                let sinAlpha = max(0, 1 - cosAlpha * cosAlpha).squareRoot()
                let phi = samples.1 * 2 * Real.pi
                let frame = ShadingFrame(x: -wcX, y: -wcY, z: -vectorCenter)
                let worldNormal = Normal(
                        sphericalDirection(
                                sinTheta: sinAlpha,
                                cosTheta: cosAlpha,
                                phi: phi,
                                frame: frame))
                let worldPoint = center + radius * Point(worldNormal)
                let interaction = SurfaceInteraction(position: worldPoint, normal: worldNormal)
                let pdf: Real = 1.0 / (2 * Real.pi * (1 - cosThetaMax))
                return (interaction, pdf)
        }

        func area(scene _: Scene) -> Real {
                return 4 * Real.pi * radius * radius
        }

        func quadratic(coeffA: Real, coeffB: Real, coeffC: Real) -> (Real, Real)? {
                let discriminant = coeffB * coeffB - 4 * coeffA * coeffC
                guard discriminant > 0 else {
                        return nil
                }
                let rootDiscriminant = discriminant.squareRoot()
                var qValue: Real = 0.0
                if coeffB < 0 {
                        qValue = -Real(0.5) * (coeffB - rootDiscriminant)
                } else {
                        qValue = -Real(0.5) * (coeffB + rootDiscriminant)
                }
                let roots = [qValue / coeffA, coeffC / qValue].sorted()
                return (roots[0], roots[1])
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout Real
        ) throws -> Bool {
                throw RenderError.unimplemented(function: #function, file: #file, line: #line, message: "")
        }

        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                let ray = getWorldToObject(scene: scene) * worldRay
                let originX = ray.origin.x
                let originY = ray.origin.y
                let originZ = ray.origin.z
                let directionX = ray.direction.x
                let directionY = ray.direction.y
                let directionZ = ray.direction.z
                let coeffA = directionX * directionX + directionY * directionY + directionZ * directionZ
                let coeffB = 2 * (directionX * originX + directionY * originY + directionZ * originZ)
                let coeffC = originX * originX + originY * originY + originZ * originZ - radius * radius

                guard let tValues = quadratic(coeffA: coeffA, coeffB: coeffB, coeffC: coeffC) else {
                        return nil
                }
                guard tValues.0 < tHit && tValues.1 > 0 else {
                        return nil
                }
                var shapeHit = tValues.0
                if shapeHit <= 0 {
                        shapeHit = tValues.1
                        if shapeHit > tHit {
                                return nil
                        }
                }
                var pHit = ray.getPointFor(parameter: shapeHit)
                pHit *= radius / distance(pHit, origin)
                if pHit.x == 0 && pHit.y == 0 { pHit.x = 0.00001 * radius }
                let normal = normalized(Normal(point: pHit))

                let phiMax = 2 * Real.pi
                let dpdu = Vector(x: -phiMax * pHit.y, y: phiMax * pHit.x, z: 0)

                let uvCoordinates = Point2f()
                let localInteraction = SurfaceInteraction(
                        position: pHit,
                        normal: normal,
                        shadingNormal: normal,
                        outgoing: -ray.direction,
                        dpdu: dpdu,
                        uvCoordinates: uvCoordinates)
                tHit = shapeHit
                return objectToWorld * localInteraction
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let objectToWorld: Transform
        let radius: Real
}

extension Sphere {
        static func create(objectToWorld: Transform, parameters: ParameterDictionary) throws -> ShapeType {
        let radius = try parameters.findOneReal(called: "radius", else: 1.0)
        return .sphere(Sphere(radius: radius, objectToWorld: objectToWorld))
}
}
