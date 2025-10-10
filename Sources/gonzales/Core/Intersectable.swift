/// A type that can be intersected by a ray.

protocol Intersectable {
        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction) throws
}

enum IntersectablePrimitive: Intersectable, Sendable {
        case geometricPrimitive(GeometricPrimitive)
        case triangle(Triangle)
        case transformedPrimitive(TransformedPrimitive)
        case areaLight(AreaLight)

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                switch self {
                case .areaLight(let areaLight):
                        return try areaLight.intersect(
                                ray: ray,
                                tHit: &tHit)
                case .geometricPrimitive(let geometricPrimitive):
                        return try geometricPrimitive.intersect(
                                ray: ray,
                                tHit: &tHit)
                case .triangle(let triangle):
                        return try triangle.intersect(
                                ray: ray,
                                tHit: &tHit)
                case .transformedPrimitive(let transformedPrimitive):
                        return try transformedPrimitive.intersect(
                                ray: ray,
                                tHit: &tHit)
                }
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                switch self {
                case .areaLight(let areaLight):
                        try areaLight.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                case .geometricPrimitive(let geometricPrimitive):
                        try geometricPrimitive.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                case .triangle(let triangle):
                        try triangle.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                case .transformedPrimitive(let transformedPrimitive):
                        try transformedPrimitive.intersect(
                                ray: ray,
                                tHit: &tHit,
                                interaction: &interaction)
                }
        }
}
