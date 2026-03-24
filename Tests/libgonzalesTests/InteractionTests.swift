import Testing

@testable import libgonzales

@Suite struct InteractionTests {

        @Test func offsetRayOriginSameDirection() {
                let point = Point(x: 0, y: 0, z: 0)
                let normal = Normal(x: 0, y: 0, z: 1)
                let direction = Vector(x: 0, y: 0, z: 1)
                let result = offsetRayOrigin(point: point, normal: normal, direction: direction)
                // Should offset along normal since dot(direction, normal) > 0
                #expect(result.z > point.z)
        }

        @Test func offsetRayOriginOppositeDirection() {
                let point = Point(x: 0, y: 0, z: 0)
                let normal = Normal(x: 0, y: 0, z: 1)
                let direction = Vector(x: 0, y: 0, z: -1)
                let result = offsetRayOrigin(point: point, normal: normal, direction: direction)
                // Should offset against normal since dot(direction, normal) < 0
                #expect(result.z < point.z)
        }

        @Test func spawnRayFromPointToTarget() {
                let from = Point(x: 0, y: 0, z: 0)
                let target = Point(x: 10, y: 0, z: 0)
                let (ray, _) = spawnRay(from: from, target: target)
                // Ray direction should point roughly towards target
                #expect(ray.direction.x > 0)
        }

        @Test func spawnRayTHitLessThanOne() {
                let from = Point(x: 0, y: 0, z: 0)
                let target = Point(x: 10, y: 0, z: 0)
                let (_, tHit) = spawnRay(from: from, target: target)
                // tHit should be slightly less than 1 to avoid self-intersection
                #expect(tHit < 1.0)
                #expect(tHit > 0.99)
        }
}
