struct CoordinateFrame {

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

        let x: Vector
        let y: Vector
        let z: Vector
}
