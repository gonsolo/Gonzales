/// A type that can be intersected by a ray.

protocol Intersectable {
        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX) throws -> IntersectablePrimitive?
        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction) throws
        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX) throws -> SurfaceInteraction
}

enum IntersectablePrimitive: Sendable {
        case curve(Curve)
        case embreeCurve(EmbreeCurve)
        case disk(Disk)
        case geometricPrimitive(GeometricPrimitive)
        case triangle(Triangle)
        case sphere(Sphere)
        case transformedPrimitive(TransformedPrimitive)
        case areaLight(AreaLight)

        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX,
        ) throws -> IntersectablePrimitive? {
                switch self {
                case .areaLight(let areaLight):
                        return try areaLight.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .geometricPrimitive(let geometricPrimitive):
                        return try geometricPrimitive.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .curve(let curve):
                        return try curve.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .embreeCurve(let embreeCurve):
                        return try embreeCurve.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .sphere(let sphere):
                        return try sphere.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .disk(let disk):
                        return try disk.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .triangle(let triangle):
                        return try triangle.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                case .transformedPrimitive(let transformedPrimitive):
                        return try transformedPrimitive.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                }
        }

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> SurfaceInteraction {
                switch self {
                case .triangle(let triangle):
                       return try triangle.computeInteraction(ray: ray, tHit: &tHit) 
                case .areaLight(let areaLight):
                       return try areaLight.computeInteraction(ray: ray, tHit: &tHit) 
                case .curve:
                        unimplemented()
                case .embreeCurve:
                        unimplemented()
                case .disk:
                        unimplemented()
                case .geometricPrimitive(let geometricPrimitive):
                       return try geometricPrimitive.computeInteraction(ray: ray, tHit: &tHit) 
                case .sphere:
                        unimplemented()
                case .transformedPrimitive:
                        unimplemented()
                }
        }

        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction
        //) throws {
        //        switch self {
        //        case .areaLight(let areaLight):
        //                try areaLight.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .geometricPrimitive(let geometricPrimitive):
        //                try geometricPrimitive.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .disk(let disk):
        //                try disk.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .curve(let curve):
        //                try curve.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .embreeCurve(let embreeCurve):
        //                try embreeCurve.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .triangle(let triangle):
        //                try triangle.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .sphere(let sphere):
        //                try sphere.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        case .transformedPrimitive(let transformedPrimitive):
        //                try transformedPrimitive.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        }
        //}
}
