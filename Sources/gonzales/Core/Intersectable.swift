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

enum IntersectablePrimitiveIntersection {
        case triangle(TriangleIntersection?)
}

enum IntersectablePrimitive: Intersectable, Sendable {
        case geometricPrimitive(GeometricPrimitive)
        case triangle(Triangle)
        case transformedPrimitive(TransformedPrimitive)
        case areaLight(AreaLight)

        func getIntersectionData(
                ray worldRay: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitiveIntersection {
                switch self {
                case .areaLight(let areaLight):
                        return try areaLight.getIntersectionData(
                                ray: worldRay,
                                tHit: &tHit)
                case .geometricPrimitive(let geometricPrimitive):
                        return try geometricPrimitive.getIntersectionData(
                                ray: worldRay,
                                tHit: &tHit)
                case .triangle(let triangle):
                        return try .triangle(
                                triangle.getIntersectionData(
                                        ray: worldRay,
                                        tHit: &tHit))
                //case .transformedPrimitive(let transformedPrimitive):
                case .transformedPrimitive:
                        unimplemented()
                //        return try transformedPrimitive.intersect(
                //                ray: ray,
                //                tHit: &tHit)
                //default:
                //        unimplemented()
                }

        }

        func computeSurfaceInteraction(
                data: IntersectablePrimitiveIntersection,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) {
                switch self {
                case .geometricPrimitive(let geometricPrimitive):
                        return geometricPrimitive.computeSurfaceInteraction(
                                data: data,
                                worldRay: worldRay,
                                interaction: &interaction)
                case .triangle(let triangle):
                        switch data {
                        case .triangle(let triangleIntersection):
                                guard let triangleIntersection = triangleIntersection else {
                                        return
                                }
                                return triangle.computeSurfaceInteraction(
                                        data: triangleIntersection,
                                        worldRay: worldRay,
                                        interaction: &interaction)

                        }
                case .transformedPrimitive:
                        unimplemented()
                case .areaLight(let areaLight):
                        return areaLight.computeSurfaceInteraction(
                                data: data,
                                worldRay: worldRay,
                                interaction: &interaction)
                }
        }

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
