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

                // Compute time where ray hits z=0 plane
                let rayTime = -ray.origin.z / ray.direction.z
                guard rayTime > 0 && rayTime < tHit else { return nil }

                // Check if hit point is within the disk radius
                let pHit = ray.getPointFor(parameter: rayTime)
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
                tHit = rayTime
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
                let (localX, localY) = concentricSampleDisk(sampleU: samples.0, sampleV: samples.1)
                let localPosition = Point(x: localX * radius, y: localY * radius, z: 0)
                let worldNormal = normalized(objectToWorld * Normal(x: 0, y: 0, z: 1))
                let worldPosition = objectToWorld * localPosition
                let interaction = SurfaceInteraction(position: worldPosition, normal: worldNormal)
                let pdf = 1 / (Real.pi * radius * radius)
                return (interaction, pdf)
        }

        // Maps a unit square sample to a unit disk using Shirley's concentric mapping.
        private func concentricSampleDisk(sampleU: Real, sampleV: Real) -> (Real, Real) {
                let sampleX = 2 * sampleU - 1
                let sampleY = 2 * sampleV - 1
                guard sampleX != 0 || sampleY != 0 else { return (0, 0) }
                let (radiusLocal, theta): (Real, Real)
                if abs(sampleX) > abs(sampleY) {
                        radiusLocal = sampleX
                        theta = (Real.pi / 4) * (sampleY / sampleX)
                } else {
                        radiusLocal = sampleY
                        theta = (Real.pi / 2) - (Real.pi / 4) * (sampleX / sampleY)
                }
                return (radiusLocal * cos(theta), radiusLocal * sin(theta))
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
