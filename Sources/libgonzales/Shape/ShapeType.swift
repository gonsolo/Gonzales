enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)

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
                }
        }

        func objectBound(scene: Scene) throws -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.objectBound(scene: scene)
                case .sphere(let sphere):
                        return sphere.objectBound(scene: scene)
                case .disk(let disk):
                        return try disk.objectBound(scene: scene)
                case .curve(let curve):
                        return curve.objectBound(scene: scene)
                }
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                switch self {
                case .triangle(let triangle):
                        return triangle.worldBound(scene: scene)
                case .sphere(let sphere):
                        return sphere.worldBound(scene: scene)
                case .disk(let disk):
                        return try disk.worldBound(scene: scene)
                case .curve(let curve):
                        return curve.worldBound(scene: scene)
                }
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                switch self {
                case .triangle(let triangle):
                        return try triangle.getIntersectionData(
                                scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                default:
                        throw RenderError.unimplemented(function: #function, file: #filePath, line: #line, message: "")
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                switch self {
                case .triangle(let triangle):
                        return triangle.computeSurfaceInteraction(
                                scene: scene,
                                data: data,
                                worldRay: worldRay)
                default:
                        throw RenderError.unimplemented(function: #function, file: #filePath, line: #line, message: "")
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {

                switch self {
                case .triangle(let triangle):
                        return try triangle.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .sphere(let sphere):
                        return try sphere.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .disk(let disk):
                        return try disk.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .curve(let curve):
                        return try curve.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func sample(samples: TwoRandomVariables, scene: Scene) throws -> (
                interaction: SurfaceInteraction, pdf: Real
        ) {
                switch self {
                case .triangle(let triangle):
                        return triangle.sample(samples: samples, scene: scene)
                case .sphere(let sphere):
                        return sphere.sample(samples: samples, scene: scene)
                case .disk(let disk):
                        return try disk.sample(samples: samples, scene: scene)
                case .curve(let curve):
                        return try curve.sample(samples: samples, scene: scene)
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, scene: Scene) throws -> (SurfaceInteraction, Real) {
                switch self {
                case .triangle(let triangle):
                        return try triangle.sample(point: point, samples: samples, scene: scene)
                case .sphere(let sphere):
                        return try sphere.sample(point: point, samples: samples, scene: scene)
                case .disk(let disk):
                        return try disk.sample(point: point, samples: samples, scene: scene)
                case .curve(let curve):
                        return try curve.sample(point: point, samples: samples, scene: scene)
                }
        }

        func area(scene: Scene) throws -> Area {
                switch self {
                case .triangle(let triangle):
                        return triangle.area(scene: scene)
                case .sphere(let sphere):
                        return sphere.area(scene: scene)
                case .disk(let disk):
                        return try disk.area(scene: scene)
                case .curve(let curve):
                        return try curve.area(scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: I
        ) throws -> Real {
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
                }
        }
}
