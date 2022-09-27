///        A type that provides samples points.
protocol Sampler {
        func get1D() -> FloatX
        func get2D() -> Point2F
        func clone() -> Sampler
        func getCameraSample(pixel: Point2I) -> CameraSample

        var samplesPerPixel: Int { get set }
}

extension Sampler {

        func getCameraSample(pixel: Point2I) -> CameraSample {
                var cameraSample = CameraSample()
                let offset = get2D()
                cameraSample.film = Point2F(from: pixel) + offset
                cameraSample.lens = get2D()
                return cameraSample
        }
}
