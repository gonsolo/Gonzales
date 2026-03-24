import Testing

@testable import libgonzales

@Suite struct RayTests {

        @Test func getPointForParameterZero() {
                let ray = Ray(
                        origin: Point(x: 1, y: 2, z: 3),
                        direction: Vector(x: 0, y: 0, z: -1))
                let result = ray.getPointFor(parameter: 0)
                #expect(abs(result.x - 1) <= 1e-6)
                #expect(abs(result.y - 2) <= 1e-6)
                #expect(abs(result.z - 3) <= 1e-6)
        }

        @Test func getPointForParameterOne() {
                let origin = Point(x: 1, y: 2, z: 3)
                let direction = Vector(x: 0, y: 0, z: -1)
                let ray = Ray(origin: origin, direction: direction)
                let result = ray.getPointFor(parameter: 1)
                #expect(abs(result.x - (origin.x + direction.x)) <= 1e-6)
                #expect(abs(result.y - (origin.y + direction.y)) <= 1e-6)
                #expect(abs(result.z - (origin.z + direction.z)) <= 1e-6)
        }

        @Test func getPointForParameterArbitrary() {
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 0),
                        direction: Vector(x: 1, y: 0, z: 0))
                let result = ray.getPointFor(parameter: 5)
                #expect(abs(result.x - 5) <= 1e-6)
                #expect(abs(result.y) <= 1e-6)
                #expect(abs(result.z) <= 1e-6)
        }

        @Test func inverseDirectionCorrect() {
                let ray = Ray(
                        origin: Point(x: 0, y: 0, z: 0),
                        direction: Vector(x: 2, y: 4, z: 0.5))
                #expect(abs(ray.inverseDirection.x - 0.5) <= 1e-5)
                #expect(abs(ray.inverseDirection.y - 0.25) <= 1e-5)
                #expect(abs(ray.inverseDirection.z - 2.0) <= 1e-5)
        }

        @Test func defaultRayOriginAndDirection() {
                let ray = Ray()
                #expect(abs(ray.origin.x) <= 1e-6)
                #expect(abs(ray.origin.y) <= 1e-6)
                #expect(abs(ray.origin.z) <= 1e-6)
                #expect(abs(ray.direction.x) <= 1e-6)
                #expect(abs(ray.direction.y) <= 1e-6)
                #expect(abs(ray.direction.z - 1) <= 1e-6)
        }
}
