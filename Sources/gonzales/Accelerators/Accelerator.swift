protocol Accelerator: Boundable, Intersectable {

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction) throws

        func worldBound() -> Bounds3f
}
