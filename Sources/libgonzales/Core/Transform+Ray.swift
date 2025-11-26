extension Transform {

        @inline(__always)
        static func * (transform: Transform, ray: Ray) -> Ray {
                return Ray(
                        origin: transform * ray.origin,
                        direction: transform * ray.direction,
                        cameraSample: ray.cameraSample
                )
        }
}
