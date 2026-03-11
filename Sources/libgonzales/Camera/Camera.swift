/// A camera through which the scene is seen.
///
/// It generates viewing rays per pixel into the scene
/// and records the computed radiance on the film.

protocol Camera: Sendable {

        func generateRay(cameraSample: CameraSample) -> Ray

        func getSampleBounds() -> Bounds2i

        var film: Film { get }
}

extension Camera {

        func getSampleBounds() -> Bounds2i {
                return film.getSampleBounds()
        }

}
