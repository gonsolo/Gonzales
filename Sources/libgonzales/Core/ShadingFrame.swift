public struct ShadingFrame: Sendable {

        let tangent: Vector
        let bitangent: Vector
        let normal: Normal

        public init(tangent: Vector, bitangent: Vector, normal: Normal) {
                self.tangent = tangent
                self.bitangent = bitangent
                self.normal = normal
        }

}

extension ShadingFrame {

        public init() {
                tangent = nullVector
                bitangent = nullVector
                normal = upNormal
        }

        public init(tangent: Vector, bitangent: Vector) {
                self.init(tangent: tangent, bitangent: bitangent, normal: Normal(cross(tangent, bitangent)))
        }

        public init(normal: Normal) {
                print("initializing ShadingFrame from normal: ", normal)
                let n = Vector(normal: normalized(normal))

                // We use the same variable names (x, y, z) as the pbrt function internally
                let z = n
                var x: Vector
                var y: Vector

                // PBRT's CoordinateSystem logic (Adapted for Swift)
                // Note: pbrt uses Float as the type for its intermediate calculations

                let sign: FloatX = (z.z >= 0) ? 1.0 : -1.0

                // Safe reciprocal: -1 / (sign + v1.z)
                // We assume Sqr(v) = v * v is available
                let a = -1.0 / (sign + z.z)
                let b = z.x * z.y * a

                // pbrt's v2 (which is our Tangent, x)
                x = Vector(
                        x: 1.0 + sign * z.x * z.x * a,
                        y: sign * b,
                        z: -sign * z.x
                )

                // pbrt's v3 (which is our Bitangent, y)
                y = Vector(
                        x: b,
                        y: sign + z.y * z.y * a,
                        z: -z.y
                )

                // Set the frame (t=x, b=y, n=z)
                print("resulting ShadingFrame: ", x, y, z)
                self.init(tangent: x, bitangent: y, normal: Normal(z))

                // OPTIONAL: Sanity check (n = t x b) for right-handedness.
                let check = cross(x, y)
                assert(dot(check, z) > 0.9999, "ShadingFrame is not right-handed")
        }

        //public init(normal: Normal) {
        //        print("initializing ShadingFrame from normal: ", normal)
        //        let n = normalized(normal)
        //        var a: Normal
        //        if abs(n.x) < abs(n.y) {
        //                a = (abs(n.x) < abs(n.z)) ? Normal(x: 1, y: 0, z: 0) : Normal(x: 0, y: 0, z: 1)
        //        } else {
        //                a = (abs(n.y) < abs(n.z)) ? Normal(x: 0, y: 1, z: 0) : Normal(x: 0, y: 0, z: 1)
        //        }
        //        let c = cross(a, n)
        //        let t = Vector(normal: normalized(c))
        //        let b = -cross(t, Vector(normal: n))
        //        print("resulting ShadingFrame: ", t, b, n)
        //        self.init(tangent: t, bitangent: b, normal: n)
        //}
}

extension ShadingFrame {

        public func worldToLocal(world: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: dot(world, tangent),
                                y: dot(world, bitangent),
                                z: dot(world, normal)))
        }

        public func localToWorld(local: Vector) -> Vector {
                let a = local.x * tangent
                let b = local.y * bitangent 
                let c = Vector(normal: local.z * normal)
                return a + b + c
        }
}

