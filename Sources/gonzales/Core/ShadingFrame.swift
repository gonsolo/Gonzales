struct ShadingFrame {

        init() {
                x = nullVector
                y = nullVector
                z = nullVector
        }

        init(x: Vector, y: Vector, z: Vector) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(x: Vector, y: Vector) {
                self.init(x: x, y: y, z: cross(x, y))
        }

        func worldToLocal(world: Vector) -> Vector {
                return normalized(
                        Vector(
                                x: dot(world, y),
                                y: dot(world, z),
                                z: dot(world, x)))
        }

        func localToWorld(local: Vector) -> Vector {
                let vx = y.x * local.x + z.x * local.y + x.x * local.z
                let vy = y.y * local.x + z.y * local.y + x.y * local.z
                let vz = y.z * local.x + z.z * local.y + x.z * local.z
                return normalized(Vector(x: vx, y: vy, z: vz))
        }

        let x: Vector
        let y: Vector
        let z: Vector
}
