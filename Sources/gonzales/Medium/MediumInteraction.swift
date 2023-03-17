protocol PhaseFunction {

        func samplePhase(wo: Vector, sampler: Sampler) -> (FloatX, Vector)
}

final class HenyeyGreenstein: PhaseFunction {

        func samplePhase(wo: Vector, sampler: Sampler) -> (FloatX, Vector) {

                return (1, up)
        }
}

struct MediumInteraction: Interaction {

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2F()
        var wo = Vector()

        var phase = HenyeyGreenstein()
}
