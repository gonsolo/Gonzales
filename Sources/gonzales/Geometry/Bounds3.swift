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
                get {
                        switch index {
                        case 0: return pMin
                        case 1: return pMax
                        default: return pMin
                        }
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
                        Point3(x: pMax.x, y: pMax.y, z: pMax.z),
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
                for i in 0..<3 {
                        if pMax[i] > pMin[i] { o[i] /= pMax[i] - pMin[i] }
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

// Global function to compute the component-wise minimum of two Vector3s
// This function must be visible to the code that calls min(t0_v, t1_v).
@inline(__always)
func min(_ lhs: Vector3, _ rhs: Vector3) -> Vector3 {
        // Manually compare and select the minimum component for each axis.
        // This assumes your Vector3 exposes x, y, z properties.
        let minX = min(lhs.x, rhs.x)
        let minY = min(lhs.y, rhs.y)
        let minZ = min(lhs.z, rhs.z)

        // Return a new Vector3 constructed from these three scalar minimums.
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
                // 1. Calculate intersection t-values for all 3 axes simultaneously (SIMD)
                // t0_v = (pMin - ray.origin) * ray.inverseDirection
                let t0_v = (pMin - ray.origin) * ray.inverseDirection
                let t1_v = (pMax - ray.origin) * ray.inverseDirection

                // 2. Determine component-wise tNear and tFar (SIMD)
                // The min/max operations on the vector handle the 'swap' logic branchlessly.
                let tNear_v = min(t0_v, t1_v)
                var tFar_v = max(t0_v, t1_v)

                // 3. Apply floating-point error correction (SIMD/Scalar)
                // Ensure `gamma(n: 3)` is a highly optimized, constant-folding function.
                let errorCorrection: FloatX = 1.0 + 2.0 * gamma(n: 3)
                tFar_v *= errorCorrection

                // 4. Horizontal Reduction: Find the overall tNear and tFar (Scalar operations on SIMD components)
                // The ray hits the box at the largest tNear and leaves at the smallest tFar.

                // Find tNear: Max of (tNear_v.x, tNear_v.y, tNear_v.z)
                var tNear = max(tNear_v.x, tNear_v.y)
                tNear = max(tNear, tNear_v.z)  // This is the horizontal max reduction (max component)

                // Find tFar: Min of (tFar_v.x, tFar_v.y, tFar_v.z)
                var tFar = min(tFar_v.x, tFar_v.y)
                tFar = min(tFar, tFar_v.z)  // This is the horizontal min reduction (min component)

                // 5. Final boundary checks (Scalar)
                let t_start: FloatX = 0.0

                // Apply bounds: Max(tNear, 0) and Min(tFar, tHit)
                tNear = max(tNear, t_start)
                tFar = min(tFar, tHit)

                // Final check: does the near hit occur before the current best hit (tHit) and before the ray leaves the far plane?
                return tNear <= tFar
        }

        static func statistics() {
                print("  Bounds3:")
                // TODO
        }
}

extension Bounds3f {
        var center: Point {
                return 0.5 * self.pMin + 0.5 * self.pMax
        }
}
