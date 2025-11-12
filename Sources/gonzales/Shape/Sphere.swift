import Foundation

struct Sphere: Shape {

        init(radius: FloatX, objectToWorld: Transform) {
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

                let p0 = Point(x: -radius, y: -radius, z: -radius)
                let p1 = Point(x: +radius, y: +radius, z: +radius)

                return Bounds3f(first: p0, second: p1)
        }

        func uniformSampleSphere(samples: TwoRandomVariables) -> Point {

                let z = 1 - 2 * samples.0
                let r = max(0, 1 - z * z).squareRoot()
                let phi = 2 * FloatX.pi * samples.1
                return Point(x: r * cos(phi), y: r * sin(phi), z: z)
        }

        func sample(samples: TwoRandomVariables, scene _: Scene) -> (
                interaction: SurfaceInteraction, pdf: FloatX
        ) {

                let localPosition = radius * uniformSampleSphere(samples: samples)
                let worldNormal = normalized(objectToWorld * Normal(point: localPosition))
                let worldPosition = objectToWorld * localPosition
                let interaction = SurfaceInteraction(position: worldPosition, normal: worldNormal)
                let pdf: FloatX = 1.0
                return (interaction, pdf)
        }

        func sample(ref: any Interaction, samples: TwoRandomVariables, scene: Scene)
                -> (interaction: any Interaction, pdf: FloatX) {
                let center = objectToWorld * origin
                if distanceSquared(ref.position, center) <= radius * radius {
                        var (interaction, pdf) = sample(samples: samples, scene: scene)
                        var wi: Vector = interaction.position - ref.position
                        if lengthSquared(wi) == 0 {
                                pdf = 0
                        } else {
                                wi = normalized(wi)
                                pdf *=
                                        distanceSquared(ref.position, interaction.position)
                                        / absDot(interaction.normal, Normal(-wi))
                        }
                        if pdf.isInfinite { pdf = 0 }
                        return (interaction, pdf)
                }
                let dc = distance(ref.position, center)
                let invDc = 1 / dc
                let wc = Vector(vector: center - ref.position) * invDc
                let (wcX, wcY) = makeCoordinateSystem(from: wc)
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
                let phi = samples.1 * 2 * FloatX.pi
                let worldNormal = Normal(
                        sphericalDirection(
                                sinTheta: sinAlpha,
                                cosTheta: cosAlpha,
                                phi: phi,
                                x: -wcX,
                                y: -wcY,
                                z: -wc))
                let worldPoint = center + radius * Point(worldNormal)
                let interaction = SurfaceInteraction(position: worldPoint, normal: worldNormal)
                let pdf: FloatX = 1.0 / (2 * FloatX.pi * (1 - cosThetaMax))
                return (interaction, pdf)
        }

        func area(scene _: Scene) -> FloatX {
                return 4 * FloatX.pi * radius * radius
        }

        func quadratic(a: FloatX, b: FloatX, c: FloatX) -> (FloatX, FloatX)? {
                let discriminant = b * b - 4 * a * c
                guard discriminant > 0 else {
                        return nil
                }
                let rootDiscriminant = discriminant.squareRoot()
                var q: FloatX = 0.0
                if b < 0 {
                        q = -FloatX(0.5) * (b - rootDiscriminant)
                } else {
                        q = -FloatX(0.5) * (b + rootDiscriminant)
                }
                let ta = [q / a, c / q].sorted()
                return (ta[0], ta[1])
        }

        func intersect(
                scene _: Scene,
                ray _: Ray,
                tHit _: inout FloatX
        ) throws -> Bool {
                unimplemented()
        }

        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                let empty = { (_: Int) in
                        return
                }

                let ray = getWorldToObject(scene: scene) * worldRay
                let ox = ray.origin.x
                let oy = ray.origin.y
                let oz = ray.origin.z
                let dx = ray.direction.x
                let dy = ray.direction.y
                let dz = ray.direction.z
                let a = dx * dx + dy * dy + dz * dz
                let b = 2 * (dx * ox + dy * oy + dz * oz)
                let c = ox * ox + oy * oy + oz * oz - radius * radius

                guard let t = quadratic(a: a, b: b, c: c) else {
                        return empty(#line)
                }
                guard t.0 < tHit && t.1 > 0 else {
                        return empty(#line)
                }
                var shapeHit = t.0
                if shapeHit <= 0 {
                        shapeHit = t.1
                        if shapeHit > tHit {
                                return empty(#line)
                        }
                }
                var pHit = ray.getPointFor(parameter: shapeHit)
                pHit *= radius / distance(pHit, origin)
                if pHit.x == 0 && pHit.y == 0 { pHit.x = 0.00001 * radius }
                let normal = normalized(Normal(point: pHit))

                let phiMax = 2 * FloatX.pi
                let dpdu = Vector(x: -phiMax * pHit.y, y: phiMax * pHit.x, z: 0)

                let uv = Point2f()
                let localInteraction = SurfaceInteraction(
                        valid: true,
                        position: pHit,
                        normal: normal,
                        shadingNormal: normal,
                        wo: -ray.direction,
                        dpdu: dpdu,
                        uv: uv)
                interaction = objectToWorld * localInteraction
                tHit = shapeHit
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let objectToWorld: Transform
        let radius: FloatX
}

func createSphere(objectToWorld: Transform, parameters: ParameterDictionary) throws -> ShapeType {
        let radius = try parameters.findOneFloatX(called: "radius", else: 1.0)
        return .sphere(Sphere(radius: radius, objectToWorld: objectToWorld))
}
