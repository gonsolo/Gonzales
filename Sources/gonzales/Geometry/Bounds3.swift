public struct Bounds3<T: Comparable> {

        init(first: Point3<T>, second: Point3<T>) {
                self.pMin = Point3(
                        x: min(first.x, second.x),
                        y: min(first.y, second.y),
                        z: min(first.z, second.z))
                self.pMax = Point3(
                        x: max(first.x, second.x),
                        y: max(first.y, second.y),
                        z: max(first.z, second.z))
        }

        subscript(index: Int) -> Point3<T> {
                get {
                        switch index {
                        case 0: return pMin
                        case 1: return pMax
                        default: return pMin
                        }
                }
        }

        mutating func add(point: Point3<T>) {
                if point.x < pMin.x { pMin.x = point.x }
                if point.y < pMin.y { pMin.y = point.y }
                if point.z < pMin.z { pMin.z = point.z }
                if point.x > pMax.x { pMax.x = point.x }
                if point.y > pMax.y { pMax.y = point.y }
                if point.z > pMax.z { pMax.z = point.z }
        }

        var points: [Point3<T>] {
                return [
                        Point3(x: pMin.x, y: pMin.y, z: pMin.z),
                        Point3(x: pMin.x, y: pMin.y, z: pMax.z),
                        Point3(x: pMin.x, y: pMax.y, z: pMin.z),
                        Point3(x: pMin.x, y: pMax.y, z: pMax.z),
                        Point3(x: pMax.x, y: pMin.y, z: pMin.z),
                        Point3(x: pMax.x, y: pMin.y, z: pMax.z),
                        Point3(x: pMax.x, y: pMax.y, z: pMin.z),
                        Point3(x: pMax.x, y: pMax.y, z: pMax.z),
                ]
        }

        var pMin: Point3<T>
        var pMax: Point3<T>
}

extension Bounds3: CustomStringConvertible {
        public var description: String { return "Bounds3 [ \(pMin) - \(pMax) ]" }
}

public typealias Bounds3f = Bounds3<FloatX>

extension Bounds3 where T == FloatX {

        static var counter = 0

        init() {
                Self.counter += 1
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

        func diagonal() -> Vector3<T> {
                return Vector3(point: pMax - pMin)
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

extension Bounds3 where T == FloatX {

        static var intersections = 0

        func intersects(ray: Ray, tHit: inout FloatX) -> Bool {
                Self.intersections += 1
                var t0: FloatX = 0.0
                var t1 = tHit
                for i in 0..<3 {
                        let invRayDir = 1 / ray.direction[i]
                        var tNear = (pMin[i] - ray.origin[i]) * invRayDir
                        var tFar = (pMax[i] - ray.origin[i]) * invRayDir
                        if tNear > tFar {
                                swap(&tNear, &tFar)
                        }
                        tFar *= 1 + 2 * gamma(n: 3)
                        t0 = tNear > t0 ? tNear : t0
                        t1 = tFar < t1 ? tFar : t1
                        if t0 > t1 {
                                return false
                        }
                }
                return true
        }

        static func statistics() {
                print("  Bounds3:")
                print("    Generated:\t\t\t\t\t\t\t\t\(counter)")
                print("    Intersections:\t\t\t\t\t\t\t\(intersections)")
        }
}

extension Bounds3f {
        var center: Point {
                return 0.5 * self.pMin + 0.5 * self.pMax
        }
}
