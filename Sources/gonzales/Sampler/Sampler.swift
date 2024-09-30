typealias RandomVariable = FloatX
typealias TwoRandomVariables = (RandomVariable, RandomVariable)
typealias ThreeRandomVariables = (RandomVariable, RandomVariable, RandomVariable)

///        A type that provides samples points.
protocol Sampler: Sendable {

        func get1D() async -> RandomVariable
        func get2D() async -> TwoRandomVariables
        func get3D() async -> ThreeRandomVariables
        func clone() async -> Sampler
        func getCameraSample(pixel: Point2I) async -> CameraSample

        var samplesPerPixel: Int { get }
}

extension Sampler {

        func getCameraSample(pixel: Point2I) async -> CameraSample {
                return await CameraSample(
                        film: (
                                FloatX(pixel.x) + get1D(),
                                FloatX(pixel.y) + get1D()
                        ),
                        lens: get2D())
        }
}
