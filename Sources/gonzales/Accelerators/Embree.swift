import embree

class Embree {

        init(primitives: inout [Boundable & Intersectable]) {
                print("embree init")
                embreeInit()
                for primitive in primitives {
                        if let geometricPrimitive = primitive as? GeometricPrimitive {
                                print("geometricPrimitive")
                                if let triangle = geometricPrimitive.shape as? Triangle {
                                        print("triangle")
                                        geometry(triangle: triangle)
                                }
                        }
                }
        }

        deinit {
                embreeDeinit()
        }

        func commit() {
                embreeCommit()
        }

        func geometry(triangle: Triangle) {
                let points = triangle.getLocalPoints()
                let a = points.0
                let b = points.1
                let c = points.2
                embreeGeometry(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
        }

        func intersect() {
                embreeIntersect()
        }
}
