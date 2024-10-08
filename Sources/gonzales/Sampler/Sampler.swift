typealias RandomVariable = FloatX
typealias TwoRandomVariables = (RandomVariable, RandomVariable)
typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)

///        A type that provides samples points.
protocol Sampler: Sendable {

        func get1D() -> RandomVariable
        func get2D() -> TwoRandomVariables
        func get3D() -> ThreeRandomVariables
        func clone() async -> Sampler
        func getCameraSample(pixel: Point2i) async -> CameraSample

        var samplesPerPixel: Int { get }
}

extension Sampler {

        func getCameraSample(pixel: Point2i) async -> CameraSample {
                return CameraSample(
                        film: (
                                FloatX(pixel.x) + get1D(),
                                FloatX(pixel.y) + get1D()
                        ),
                        lens: get2D())
        }
}
