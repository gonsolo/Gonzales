//protocol Interaction: Sendable {
//
//        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX)
//        func spawnRay(inDirection direction: Vector) -> Ray
//
//        func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum
//        func sampleDistributionFunction(sampler: RandomSampler) -> BsdfSample
//        func evaluateProbabilityDensity(wi: Vector) -> FloatX
//
//        var dpdu: Vector { get }
//        var faceIndex: Int { get }
//        var normal: Normal { get }
//        var position: Point { get }
//        var shadingNormal: Normal { get }
//        var uv: Point2f { get }
//        var wo: Vector { get }
//}
//
//extension Interaction {
//
//        func spawnRay(inDirection direction: Vector) -> Ray {
//                let origin = offsetRayOrigin(point: position, direction: direction)
//                return Ray(origin: origin, direction: direction)
//        }
//
//        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
//                let origin = offsetRayOrigin(point: position, direction: to - position)
//                let direction: Vector = to - origin
//                return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
//        }
//
//        func offsetRayOrigin(point: Point, direction: Vector) -> Point {
//                let epsilon: FloatX = 0.0001
//                return Point(point + epsilon * direction)
//        }
//}

// InteractionType.swift

enum InteractionType: Sendable {

    // Wrap the two concrete interaction types
    case surface(SurfaceInteraction)
    case medium(MediumInteraction)

    // MARK: - Required Properties (Dispatch)

    var dpdu: Vector {
        switch self {
        case .surface(let i): return i.dpdu
        case .medium: return nullVector
        }
    }

    var faceIndex: Int {
        switch self {
        case .surface(let i): return i.faceIndex
        case .medium: return 0
        }
    }

    var normal: Normal {
        switch self {
        case .surface(let i): return i.normal
        case .medium(let i): return i.normal
        }
    }

    var position: Point {
        switch self {
        case .surface(let i): return i.position
        case .medium(let i): return i.position
        }
    }

    var shadingNormal: Normal {
        switch self {
        case .surface(let i): return i.shadingNormal
        case .medium: return zeroNormal
        }
    }

    var uv: Point2f {
        switch self {
        case .surface(let i): return i.uv
        case .medium: return Point2f(x: 0, y: 0)
        }
    }

    var wo: Vector {
        switch self {
        case .surface(let i): return i.wo
        case .medium(let i): return i.wo
        }
    }

    // MARK: - Required Methods (Dispatch)

    func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum {
        switch self {
        case .surface(let i): return i.evaluateDistributionFunction(wi: wi)
        case .medium(let i): return i.evaluateDistributionFunction(wi: wi)
        }
    }

    // Since you defined sampler as RandomSampler (a concrete type), no enum conversion is needed here.
    func sampleDistributionFunction(sampler: RandomSampler) -> BsdfSample {
        switch self {
        case .surface(let i): return i.sampleDistributionFunction(sampler: sampler)
        case .medium(let i): return i.sampleDistributionFunction(sampler: sampler)
        }
    }

    func evaluateProbabilityDensity(wi: Vector) -> FloatX {
        switch self {
        case .surface(let i): return i.evaluateProbabilityDensity(wi: wi)
        case .medium(let i): return i.evaluateProbabilityDensity(wi: wi)
        }
    }

    // MARK: - Migrated Extension Logic

    private func offsetRayOrigin(point: Point, direction: Vector) -> Point {
        let epsilon: FloatX = 0.0001
        return Point(point + epsilon * direction)
    }

    func spawnRay(inDirection direction: Vector) -> Ray {
        let origin = offsetRayOrigin(point: position, direction: direction)
        return Ray(origin: origin, direction: direction)
    }

    func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
        let origin = offsetRayOrigin(point: position, direction: to - position)
        let direction: Vector = to - origin
        // Assuming 'shadowEpsilon' is a globally accessible constant
        return (Ray(origin: origin, direction: direction), FloatX(1.0) - shadowEpsilon)
    }
}
