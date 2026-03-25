import Foundation

struct Disk: Shape {

        init(objectToWorld: Transform, radius: Distance) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        func objectBound(scene _: Scene) -> Bounds3f {
                return Bounds3f(
                        first: Point(x: -radius, y: -radius, z: 0),
                        second: Point(x: radius, y: radius, z: 0))
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return objectToWorld * objectBound(scene: scene)
        }

        // Ray-disk intersection: intersect ray with the z=0 plane,
        // then check if the hit point is within the disk radius.
        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                let ray = getWorldToObject(scene: scene) * worldRay

                // Reject rays parallel to the disk plane
                guard ray.direction.z != 0 else { return nil }

                // Compute t where ray hits z=0 plane
                let t = -ray.origin.z / ray.direction.z
                guard t > 0 && t < tHit else { return nil }

                // Check if hit point is within the disk radius
                let pHit = ray.getPointFor(parameter: t)
                let dist2 = pHit.x * pHit.x + pHit.y * pHit.y
                guard dist2 <= radius * radius else { return nil }

                // Compute surface interaction
                let normal = Normal(x: 0, y: 0, z: 1)
                let phi = atan2(pHit.y, pHit.x)
                let dpdu = Vector(x: -pHit.y, y: pHit.x, z: 0)
                let hitRadius = dist2.squareRoot()
                let uvCoordinates = Point2f(
                        x: phi / (2 * Real.pi),
                        y: 1 - hitRadius / radius)

                let interaction = SurfaceInteraction(
                        position: pHit,
                        normal: normal,
                        shadingNormal: normal,
                        outgoing: -ray.direction,
                        dpdu: dpdu,
                        uvCoordinates: uvCoordinates)
                tHit = t
                return objectToWorld * interaction
        }

        func intersect(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real
        ) -> Bool {
                return intersect(scene: scene, ray: worldRay, tHit: &tHit) != nil
        }

        func area(scene _: Scene) -> Area {
                return Real.pi * radius * radius
        }

        func sample(samples: TwoRandomVariables, scene _: Scene) -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {
                // Concentric disk sampling
                let (localX, localY) = concentricSampleDisk(u: samples.0, v: samples.1)
                let localPosition = Point(x: localX * radius, y: localY * radius, z: 0)
                let worldNormal = normalized(objectToWorld * Normal(x: 0, y: 0, z: 1))
                let worldPosition = objectToWorld * localPosition
                let interaction = SurfaceInteraction(position: worldPosition, normal: worldNormal)
                let pdf = 1 / (Real.pi * radius * radius)
                return (interaction, pdf)
        }

        // Maps a unit square sample to a unit disk using Shirley's concentric mapping.
        private func concentricSampleDisk(u: Real, v: Real) -> (Real, Real) {
                let sx = 2 * u - 1
                let sy = 2 * v - 1
                guard sx != 0 || sy != 0 else { return (0, 0) }
                let (r, theta): (Real, Real)
                if abs(sx) > abs(sy) {
                        r = sx
                        theta = (Real.pi / 4) * (sy / sx)
                } else {
                        r = sy
                        theta = (Real.pi / 2) - (Real.pi / 4) * (sx / sy)
                }
                return (r * cos(theta), r * sin(theta))
        }

        public var description: String {
                return "Disk"
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        func getWorldToObject(scene _: Scene) -> Transform {
                return objectToWorld.inverse
        }

        let objectToWorld: Transform
        let radius: Distance
}

extension Disk {
        static func create(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [ShapeType] {
                let radius = try parameters.findOneReal(called: "radius", else: 1.0)
                let shape = ShapeType.disk(Disk(objectToWorld: objectToWorld, radius: radius))
                return [shape]
        }
}
