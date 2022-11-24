/// A basic interaction when a ray hits a surface.

protocol Interaction {

        var position: Point { get }
        var normal: Normal { get }
        var shadingNormal: Normal { get }
        var dpdu: Vector { get }
        var primitive: (Boundable & Intersectable)? { get }
        var uv: Point2F { get }
        var faceIndex: Int { get }

        func spawnRay(inDirection direction: Vector) -> Ray
        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX)
        func offsetRayOrigin(point: Point, direction: Vector) -> Point
}

extension Interaction {

        func spawnRay(inDirection direction: Vector) -> Ray {
                let origin = offsetRayOrigin(point: position, direction: direction)
                return Ray(origin: origin, direction: direction)
        }

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
                let origin = offsetRayOrigin(point: position, direction: to - position)
                let direction: Vector = to - origin
                return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
        }

        func offsetRayOrigin(point: Point, direction: Vector) -> Point {
                let epsilon: FloatX = 0.0001
                return Point(point + epsilon * direction)
        }
}
