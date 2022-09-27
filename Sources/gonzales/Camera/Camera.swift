/// A camera through which the scene is seen.
///
/// It generates viewing rays per pixel into the scene
/// and records the computed radiance on the film.

protocol Camera {
        func generateRay(sample: CameraSample) -> Ray
        func statistics()
        var film: Film { get }
}
