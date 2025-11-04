enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)
        case embreeCurve(EmbreeCurve)

        func getObjectToWorld(scene: Scene) -> Transform {
                switch self {
                case .triangle(let triangle):
                        return triangle.getObjectToWorld(scene: scene)
                case .sphere(let sphere):
                        return sphere.getObjectToWorld(scene: scene)
                case .disk(let disk):
                        return disk.getObjectToWorld(scene: scene)
                case .curve(let curve):
                        return curve.getObjectToWorld(scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.getObjectToWorld(scene: scene)
                }
        }

        func objectBound(scene: Scene) -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.objectBound(scene: scene)
                case .sphere(let sphere):
                        return sphere.objectBound(scene: scene)
                case .disk(let disk):
                        return disk.objectBound(scene: scene)
                case .curve(let curve):
                        return curve.objectBound(scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.objectBound(scene: scene)
                }
        }

        func worldBound(scene: Scene) -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.worldBound(scene: scene)
                case .sphere(let sphere):
                        return sphere.worldBound(scene: scene)
                case .disk(let disk):
                        return disk.worldBound(scene: scene)
                case .curve(let curve):
                        return curve.worldBound(scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.worldBound(scene: scene)
                }
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection
        ) throws -> Bool {
                switch self {
                case .triangle(let triangle):
                        return try triangle.getIntersectionData(
                                scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                default:
                        unimplemented()
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
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
                                scene: scene,
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
                        try embreeCurve.intersect(
                                scene: scene, ray: ray, tHit: &tHit, interaction: &interaction)
                }
        }

        func sample(u: TwoRandomVariables, scene: Scene) -> (interaction: SurfaceInteraction, pdf: FloatX) {
                switch self {
                case .triangle(let triangle):
                        return triangle.sample(u: u, scene: scene)
                case .sphere(let sphere):
                        return sphere.sample(u: u, scene: scene)
                case .disk(let disk):
                        return disk.sample(u: u, scene: scene)
                case .curve(let curve):
                        return curve.sample(u: u, scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.sample(u: u, scene: scene)
                }
        }

        func sample(point: Point, u: TwoRandomVariables, scene: Scene) -> (SurfaceInteraction, FloatX) {
                switch self {
                case .triangle(let triangle):
                        return triangle.sample(point: point, u: u, scene: scene)
                case .sphere(let sphere):
                        return sphere.sample(point: point, u: u, scene: scene)
                case .disk(let disk):
                        return disk.sample(point: point, u: u, scene: scene)
                case .curve(let curve):
                        return curve.sample(point: point, u: u, scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.sample(point: point, u: u, scene: scene)
                }
        }

        func area(scene: Scene) -> FloatX {
                switch self {
                case .triangle(let triangle):
                        return triangle.area(scene: scene)
                case .sphere(let sphere):
                        return sphere.area(scene: scene)
                case .disk(let disk):
                        return disk.area(scene: scene)
                case .curve(let curve):
                        return curve.area(scene: scene)
                case .embreeCurve(let embreeCurve):
                        return embreeCurve.area(scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: I
        ) throws -> FloatX {
                switch self {
                case .triangle(let triangle):
                        return try triangle.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction, from: interaction)
                case .sphere(let sphere):
                        return try sphere.probabilityDensityFor(
                                scene: scene,
                                samplingDirection: direction, from: interaction)
                case .disk(let disk):
                        return try disk.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: interaction)
                case .curve(let curve):
                        return try curve.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: interaction)
                case .embreeCurve(let embreeCurve):
                        return try embreeCurve.probabilityDensityFor(
                                scene: scene, samplingDirection: direction, from: interaction)
                }
        }
}
