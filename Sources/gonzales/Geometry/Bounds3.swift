public struct Bounds3: Sendable {

        init(first: Point3, second: Point3) {
                self.pMin = Point3(
                        x: min(first.x, second.x),
                        y: min(first.y, second.y),
                        z: min(first.z, second.z))
                self.pMax = Point3(
                        x: max(first.x, second.x),
                        y: max(first.y, second.y),
                        z: max(first.z, second.z))
        }

        subscript(index: Int) -> Point3 {
                switch index {
                case 0: return pMin
                case 1: return pMax
                default: return pMin
                }
        }

        mutating func add(point: Point3) {
                if point.x < pMin.x { pMin.x = point.x }
                if point.y < pMin.y { pMin.y = point.y }
                if point.z < pMin.z { pMin.z = point.z }
                if point.x > pMax.x { pMax.x = point.x }
                if point.y > pMax.y { pMax.y = point.y }
                if point.z > pMax.z { pMax.z = point.z }
        }

        var points: [Point3] {
                return [
                        Point3(x: pMin.x, y: pMin.y, z: pMin.z),
                        Point3(x: pMin.x, y: pMin.y, z: pMax.z),
                        Point3(x: pMin.x, y: pMax.y, z: pMin.z),
                        Point3(x: pMin.x, y: pMax.y, z: pMax.z),
                        Point3(x: pMax.x, y: pMin.y, z: pMin.z),
                        Point3(x: pMax.x, y: pMin.y, z: pMax.z),
                        Point3(x: pMax.x, y: pMax.y, z: pMin.z),
                        Point3(x: pMax.x, y: pMax.y, z: pMax.z)
                ]
        }

        var pMin: Point3
        var pMax: Point3
}

extension Bounds3: CustomStringConvertible {
        public var description: String { return "Bounds3 [ \(pMin) - \(pMax) ]" }
}

public typealias Bounds3f = Bounds3

extension Bounds3 {

        init() {
                pMin = Point3(x: FloatX.infinity, y: FloatX.infinity, z: FloatX.infinity)
                pMax = Point3(x: -FloatX.infinity, y: -FloatX.infinity, z: -FloatX.infinity)
        }

        static func * (transform: Transform, bound: Bounds3) -> Bounds3 {
                var transformedBound = Bounds3(
                        first: transform * bound.points[0],
                        second: transform * bound.points[1])
                for i in 2..<8 {
                        transformedBound.add(point: transform * bound.points[i])
                }
                return transformedBound
        }

        func getExtent() -> Vector {
                return pMax - pMin
        }

        func maximumExtent() -> Int {
                let d = diagonal()
                if d.x > d.y && d.x > d.z {
                        return 0
                } else if d.y > d.z {
                        return 1
                } else {
                        return 2
                }
        }

        func diagonal() -> Vector3 {
                return Vector3(point: pMax - pMin)
        }

        func offset(point: Point3) -> Vector3 {
                var o: Vector3 = point - pMin
                for i in 0..<3 where pMax[i] > pMin[i] {
                        o[i] /= pMax[i] - pMin[i]
                }
                return o
        }

        func surfaceArea() -> FloatX {
                let d = diagonal()
                return 2 * (d.x * d.y + d.x * d.z + d.y * d.z)
        }
}

func union(first: Bounds3f, second: Bounds3f) -> Bounds3f {
        let pMin = Point(
                x: min(first.pMin.x, second.pMin.x),
                y: min(first.pMin.y, second.pMin.y),
                z: min(first.pMin.z, second.pMin.z))
        let pMax = Point(
                x: max(first.pMax.x, second.pMax.x),
                y: max(first.pMax.y, second.pMax.y),
                z: max(first.pMax.z, second.pMax.z))
        return Bounds3f(first: pMin, second: pMax)
}

func expand(bounds: Bounds3f, by delta: FloatX) -> Bounds3f {
        return Bounds3f(
                first: bounds.pMin - Point(x: delta, y: delta, z: delta),
                second: bounds.pMax + Point(x: delta, y: delta, z: delta))
}

@inline(__always)
func min(_ lhs: Vector3, _ rhs: Vector3) -> Vector3 {
        let minX = min(lhs.x, rhs.x)
        let minY = min(lhs.y, rhs.y)
        let minZ = min(lhs.z, rhs.z)
        return Vector3(x: minX, y: minY, z: minZ)
}

// You would implement max similarly:
@inline(__always)
func max(_ lhs: Vector3, _ rhs: Vector3) -> Vector3 {
        let maxX = max(lhs.x, rhs.x)
        let maxY = max(lhs.y, rhs.y)
        let maxZ = max(lhs.z, rhs.z)

        return Vector3(x: maxX, y: maxY, z: maxZ)
}

extension Bounds3 {

        @inline(__always)
        func intersects(ray: Ray, tHit: FloatX) -> Bool {
                let tMin = (pMin - ray.origin) * ray.inverseDirection
                let tMax = (pMax - ray.origin) * ray.inverseDirection

                let tEntry = min(tMin, tMax)
                var tExit = max(tMin, tMax)

                let errorCorrection: FloatX = 1.0 + 2.0 * gamma(count: 3)
                tExit *= errorCorrection

                var tNear = max(tEntry.x, tEntry.y)
                tNear = max(tNear, tEntry.z)

                var tFar = min(tExit.x, tExit.y)
                tFar = min(tFar, tExit.z)

                tNear = max(tNear, 0.0)
                tFar = min(tFar, tHit)

                return tNear <= tFar
        }

        static func statistics() {
                print("  Bounds3:")
        }
}

extension Bounds3f {
        var center: Point {
                return 0.5 * self.pMin + 0.5 * self.pMax
        }
}
