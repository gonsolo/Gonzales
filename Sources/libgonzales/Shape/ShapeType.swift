enum ShapeType: Shape {
        case triangle(Triangle)
        case sphere(Sphere)
        case disk(Disk)
        case curve(Curve)

        func getObjectToWorld(scene: Scene) -> Transform {
                switch self {
                case .triangle(let value): return value.getObjectToWorld(scene: scene)
                case .sphere(let value): return value.getObjectToWorld(scene: scene)
                case .disk(let value): return value.getObjectToWorld(scene: scene)
                case .curve(let value): return value.getObjectToWorld(scene: scene)
                }
        }

        func objectBound(scene: Scene) throws -> Bounds3f {
                switch self {
                case .triangle(let value): return value.objectBound(scene: scene)
                case .sphere(let value): return value.objectBound(scene: scene)
                case .disk(let value): return try value.objectBound(scene: scene)
                case .curve(let value): return value.objectBound(scene: scene)
                }
        }

        func worldBound(scene: Scene) throws -> Bounds3f {
                switch self {
                case .triangle(let value): return value.worldBound(scene: scene)
                case .sphere(let value): return value.worldBound(scene: scene)
                case .disk(let value): return try value.worldBound(scene: scene)
                case .curve(let value): return value.worldBound(scene: scene)
                }
        }

        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                switch self {
                case .triangle(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                case .sphere(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                case .disk(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                case .curve(let value): return try value.getIntersectionData(scene: scene, ray: worldRay, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                switch self {
                case .triangle(let value): return value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                case .sphere(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                case .disk(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                case .curve(let value): return try value.computeSurfaceInteraction(scene: scene, data: data, worldRay: worldRay)
                }
        }

        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) throws -> SurfaceInteraction? {
                switch self {
                case .triangle(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .sphere(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .disk(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
                case .curve(let value): return try value.intersect(scene: scene, ray: ray, tHit: &tHit)
                }
        }

        func sample(samples: TwoRandomVariables, scene: Scene) throws -> (interaction: SurfaceInteraction, pdf: Real) {
                switch self {
                case .triangle(let value): return value.sample(samples: samples, scene: scene)
                case .sphere(let value): return value.sample(samples: samples, scene: scene)
                case .disk(let value): return try value.sample(samples: samples, scene: scene)
                case .curve(let value): return try value.sample(samples: samples, scene: scene)
                }
        }

        func sample(point: Point, samples: TwoRandomVariables, scene: Scene) throws -> (SurfaceInteraction, Real) {
                switch self {
                case .triangle(let value): return try value.sample(point: point, samples: samples, scene: scene)
                case .sphere(let value): return try value.sample(point: point, samples: samples, scene: scene)
                case .disk(let value): return try value.sample(point: point, samples: samples, scene: scene)
                case .curve(let value): return try value.sample(point: point, samples: samples, scene: scene)
                }
        }

        func area(scene: Scene) throws -> Area {
                switch self {
                case .triangle(let value): return value.area(scene: scene)
                case .sphere(let value): return value.area(scene: scene)
                case .disk(let value): return try value.area(scene: scene)
                case .curve(let value): return try value.area(scene: scene)
                }
        }

        func probabilityDensityFor<I: Interaction>(
                scene: Scene,
                samplingDirection direction: Vector,
                from interaction: I
        ) throws -> Real {
                switch self {
                case .triangle(let value): return try value.probabilityDensityFor(scene: scene, samplingDirection: direction, from: interaction)
                case .sphere(let value): return try value.probabilityDensityFor(scene: scene, samplingDirection: direction, from: interaction)
                case .disk(let value): return try value.probabilityDensityFor(scene: scene, samplingDirection: direction, from: interaction)
                case .curve(let value): return try value.probabilityDensityFor(scene: scene, samplingDirection: direction, from: interaction)
                }
        }
}
