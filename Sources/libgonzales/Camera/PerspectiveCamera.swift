final class PerspectiveCamera: Camera, Transformable {

        init(
                cameraToWorld: Transform,
                screenWindow: Bounds2f,
                fov: Angle,
                focalDistance: Distance,
                lensRadius: Distance,
                film: Film
        ) throws {
                self.objectToWorld = cameraToWorld
                self.fieldOfView = fov
                self.focalDistance = focalDistance
                self.lensRadius = lensRadius
                self.film = film
                let perspective = try Transform.makePerspective(fov: fov, near: 0.01, far: 1000.0)
                let resolution = film.getResolution()
                cameraTransform = try CameraTransform(
                        cameraToScreen: perspective,
                        screenWindow: screenWindow,
                        resolution: resolution
                )
        }

        private func depthOfField(ray: Ray, cameraSample: CameraSample) -> Ray {
                guard lensRadius > 0 else {
                        return ray
                }
                let lens = lensRadius * concentricSampleDisk(uSample: cameraSample.lens)
                let focalT = focalDistance / ray.direction.z
                let pFocus = ray.getPointFor(parameter: focalT)
                let origin = Point(x: lens.x, y: lens.y, z: 0)
                let direction: Vector = normalized(pFocus - ray.origin)
                return Ray(origin: origin, direction: direction, cameraSample: cameraSample)
        }

        func generateRay(cameraSample: CameraSample) -> Ray {
                let rasterPoint = Point(x: cameraSample.film.0, y: cameraSample.film.1, z: 0)
                let cameraPoint = cameraTransform.rasterToCamera * rasterPoint
                let pinholeRay = Ray(
                        origin: origin,
                        direction: normalized(Vector(point: cameraPoint)),
                        cameraSample: cameraSample
                )
                let ray = depthOfField(ray: pinholeRay, cameraSample: cameraSample)
                // numberCameraRays += 1
                return objectToWorld * ray
        }

        func getObjectToWorld(scene _: Scene) -> Transform {
                return objectToWorld
        }

        let fieldOfView: Angle
        let focalDistance: Distance
        let lensRadius: Distance
        let cameraTransform: CameraTransform
        let objectToWorld: Transform
        let film: Film
}
