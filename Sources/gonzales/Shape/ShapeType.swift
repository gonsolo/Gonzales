enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)
        case embreeCurve(EmbreeCurve)

        var objectToWorld: Transform {
                switch self {
                case .triangle(let triangle):
                        return triangle.objectToWorld
                case .sphere(let sphere):
                        return sphere.objectToWorld
                case .disk(let disk):
                        return disk.objectToWorld
                case .curve(let curve):
                        return curve.objectToWorld
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.objectToWorld
                }
        }

        func objectBound() async -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.objectBound()
                case .sphere(let sphere):
                        return sphere.objectBound()
                case .disk(let disk):
                        return disk.objectBound()
                case .curve(let curve):
                        return curve.objectBound()
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.objectBound()
                }
        }

        func worldBound() async -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.worldBound()
                case .sphere(let sphere):
                        return sphere.worldBound()
                case .disk(let disk):
                        return disk.worldBound()
                case .curve(let curve):
                        return curve.worldBound()
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.worldBound()
                }
        }

        func getIntersectionData(
                ray worldRay: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection
        ) throws -> Bool {
                switch self {
                case .triangle(let triangle):
                        return try triangle.getIntersectionData(ray: worldRay, tHit: &tHit, data: &data)
                default:
                        unimplemented()
                }
        }

        func computeSurfaceInteraction(
                data: TriangleIntersection?,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                switch self {
                case .triangle(let triangle):
                        guard let data = data else {
                                return
                        }
                        return triangle.computeSurfaceInteraction(
                                data: data,
                                worldRay: worldRay,
                                interaction: &interaction)
                default:
                        unimplemented()
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {

                switch self {
                case .triangle(let triangle):
                        try triangle.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .sphere(let sphere):
                        try sphere.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .disk(let disk):
                        try disk.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .curve(let curve):
                        try curve.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .embreeCurve(let embreeCurve):
                        try embreeCurve.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {

                switch self {
                case .triangle(let triangle):
                        try triangle.intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                case .sphere(let sphere):
                        try sphere.intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                case .disk(let disk):
                        try disk.intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                case .curve(let curve):
                        try curve.intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                case .embreeCurve(let embreeCurve):
                        try embreeCurve.intersect(scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                }
        }

        func sample(u: TwoRandomVariables) -> (interaction: SurfaceInteraction, pdf: FloatX) {
                switch self {
                case .triangle(let triangle):
                        return triangle.sample(u: u)
                case .sphere(let sphere):
                        return sphere.sample(u: u)
                case .disk(let disk):
                        return disk.sample(u: u)
                case .curve(let curve):
                        return curve.sample(u: u)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.sample(u: u)
                }
        }

        func sample(point: Point, u: TwoRandomVariables) -> (SurfaceInteraction, FloatX) {
                switch self {
                case .triangle(let triangle):
                        return triangle.sample(point: point, u: u)
                case .sphere(let sphere):
                        return sphere.sample(point: point, u: u)
                case .disk(let disk):
                        return disk.sample(point: point, u: u)
                case .curve(let curve):
                        return curve.sample(point: point, u: u)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.sample(point: point, u: u)
                }
        }

        func area() -> FloatX {
                switch self {
                case .triangle(let triangle):
                        return triangle.area()
                case .sphere(let sphere):
                        return sphere.area()
                case .disk(let disk):
                        return disk.area()
                case .curve(let curve):
                        return curve.area()
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.area()
                }
        }

        func probabilityDensityFor<I: Interaction>(
                samplingDirection direction: Vector,
                from interaction: I
        ) throws -> FloatX {
                switch self {
                case .triangle(let triangle):
                        return try triangle.probabilityDensityFor(
                                samplingDirection: direction, from: interaction)
                case .sphere(let sphere):
                        return try sphere.probabilityDensityFor(
                                samplingDirection: direction, from: interaction)
                case .disk(let disk):
                        return try disk.probabilityDensityFor(samplingDirection: direction, from: interaction)
                case .curve(let curve):
                        return try curve.probabilityDensityFor(
                                samplingDirection: direction, from: interaction)
                case .embreeCurve(let embreeCurve):
                        return try embreeCurve.probabilityDensityFor(
                                samplingDirection: direction, from: interaction)
                }
        }
}
