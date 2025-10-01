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

    func intersect(
        ray: Ray,
        tHit: inout FloatX,
        interaction: inout SurfaceInteraction) throws {
        
        switch self {
        case .triangle(let triangle):
            try triangle.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
        case .sphere(let sphere):
            try sphere.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
        case .disk(let disk):
            try disk.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
        case .curve(let curve):
            try curve.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
        case .embreeCurve(let embreeCurve):
            try embreeCurve.intersect(ray: ray, tHit: &tHit, interaction: &interaction)
        }
    }

    func sample(u: TwoRandomVariables) -> (interaction: any Interaction, pdf: FloatX) {
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

    func sample(ref: any Interaction, u: TwoRandomVariables) -> (any Interaction, FloatX) {
        switch self {
        case .triangle(let triangle):
            return triangle.sample(ref: ref, u: u)
        case .sphere(let sphere):
            return sphere.sample(ref: ref, u: u)
        case .disk(let disk):
            return disk.sample(ref: ref, u: u)
        case .curve(let curve):
            return curve.sample(ref: ref, u: u)
        case .embreeCurve(let embreeCurve):
            return embreeCurve.sample(ref: ref, u: u)
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

    func probabilityDensityFor(
        samplingDirection direction: Vector,
        from interaction: any Interaction
    ) throws -> FloatX {
        switch self {
        case .triangle(let triangle):
            return try triangle.probabilityDensityFor(samplingDirection: direction, from: interaction)
        case .sphere(let sphere):
            return try sphere.probabilityDensityFor(samplingDirection: direction, from: interaction)
        case .disk(let disk):
            return try disk.probabilityDensityFor(samplingDirection: direction, from: interaction)
        case .curve(let curve):
            return try curve.probabilityDensityFor(samplingDirection: direction, from: interaction)
        case .embreeCurve(let embreeCurve):
            return try embreeCurve.probabilityDensityFor(samplingDirection: direction, from: interaction)
        }
    }
}
