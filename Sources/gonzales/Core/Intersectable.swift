/// A type that can be intersected by a ray.

protocol Intersectable {
        func intersect(ray: Ray, tHit: inout FloatX, material: MaterialIndex) throws
                -> SurfaceInteraction
}
