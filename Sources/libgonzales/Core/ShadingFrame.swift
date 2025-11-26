public struct ShadingFrame: Sendable {

        let tangent: Vector
        let bitangent: Vector
        let normal: Normal
}

extension ShadingFrame {

        public init() {
                tangent = nullVector
                bitangent = nullVector
                normal = upNormal
        }

        init(tangent: Vector, bitangent: Vector) {
                self.init(tangent: tangent, bitangent: bitangent, normal: Normal(cross(tangent, bitangent)))
        }

        public init(normal: Normal) {
                let n = normalized(normal)
                var a: Normal
                if abs(n.x) < abs(n.y) {
                        a = (abs(n.x) < abs(n.z)) ? Normal(x: 1, y: 0, z: 0) : Normal(x: 0, y: 0, z: 1)
                } else {
                        a = (abs(n.y) < abs(n.z)) ? Normal(x: 0, y: 1, z: 0) : Normal(x: 0, y: 0, z: 1)
                }
                let c = cross(a, n)
                let t = Vector(normal: normalized(c))
                let b = cross(t, Vector(normal: n))
                self.init(tangent: t, bitangent: b, normal: n)
        }
}

extension ShadingFrame {

        func worldToLocal(world: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: dot(world, bitangent),
                                y: dot(world, normal),
                                z: dot(world, tangent)))
        }

        func localToWorld(local: Vector) -> Vector {
                let vx = bitangent.x * local.x + normal.x * local.y + tangent.x * local.z
                let vy = bitangent.y * local.x + normal.y * local.y + tangent.y * local.z
                let vz = bitangent.z * local.x + normal.z * local.y + tangent.z * local.z
                return normalized(Vector(x: vx, y: vy, z: vz))
        }

}
