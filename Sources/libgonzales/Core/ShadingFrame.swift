public struct ShadingFrame: Sendable {

        public init() {
                tangent = nullVector
                bitangent = nullVector
                normal = nullVector
        }

        init(tangent: Vector, bitangent: Vector, normal: Vector) {
                self.tangent = tangent
                self.bitangent = bitangent
                self.normal = normal
        }

        init(tangent: Vector, bitangent: Vector) {
                self.init(tangent: tangent, bitangent: bitangent, normal: cross(tangent, bitangent))
        }

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

        let tangent: Vector
        let bitangent: Vector
        let normal: Vector
}
