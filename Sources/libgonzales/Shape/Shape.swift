///        A basic geometric shape like a triangle or a sphere.
///
///        It encapsulates the basic ingredients of a shape in that it can be
///        transformed, bounded, intersected and sampled.

protocol Shape: Transformable, Boundable, Intersectable, Sampleable, Sendable {}

extension Shape {
        func getIntersectionData(
                scene: Scene,
                ray worldRay: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws -> Bool {
                var localTHit = tHit
                if (try self.intersect(scene: scene, ray: worldRay, tHit: &localTHit)) != nil {
                        data = TriangleIntersection(
                                primId: PrimId(), tValue: localTHit, barycentric0: 0, barycentric1: 0,
                                barycentric2: 0)
                        tHit = localTHit
                        return true
                }
                return false
        }

        func computeSurfaceInteraction(
                scene: Scene,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                var tHit = data.tValue + 0.0001
                return try self.intersect(scene: scene, ray: worldRay, tHit: &tHit)
        }
}
