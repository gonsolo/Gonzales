public struct ShadingFrame: Sendable {

        let x: Vector
        let y: Vector
        let z: Vector

        public init(x: Vector, y: Vector, z: Vector) {
                self.x = x
                self.y = y
                self.z = z
        }

}

extension ShadingFrame {

        public init() {
                x = Vector(x: 1, y: 0, z: 0)
                y = Vector(x: 0, y: 1, z: 0)
                z = Vector(x: 0, y: 0, z: 1)
        }

        public init(x: Vector, y: Vector) {
                self.init(x: x, y: y, z: cross(x, y))
        }

        public init(y: Vector, z: Vector) {
                self.init(x: cross(y, z), y: y, z: z)
        }

        public init(z: Vector) {

                var x: Vector
                var y: Vector
                let z = normalized(z)

                let sign: FloatX = (z.z >= 0) ? 1.0 : -1.0

                let a = -1.0 / (sign + z.z)
                let b = z.x * z.y * a

                x = Vector(
                        x: 1.0 + sign * z.x * z.x * a,
                        y: sign * b,
                        z: -sign * z.x
                )

                y = Vector(
                        x: b,
                        y: sign + z.y * z.y * a,
                        z: -z.y
                )

                self.init(x: x, y: y, z: z)

                // OPTIONAL: Sanity check (n = t x b) for right-handedness.
                let check = cross(x, y)
                assert(dot(check, z) > 0.9999, "ShadingFrame is not right-handed")
        }
}

extension ShadingFrame {

        public func worldToLocal(world: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: dot(world, x),
                                y: dot(world, y),
                                z: dot(world, z)))
        }

        public func localToWorld(local: Vector) -> Vector {
                let a = local.x * x
                let b = local.y * y
                let c = local.z * z
                return a + b + c
        }
}
