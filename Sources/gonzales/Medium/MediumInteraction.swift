struct PhaseFunction {

        func samplePhase(wo: Vector, sampler: Sampler) -> Vector {
                // TODO
                return Vector()
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

        var phase = PhaseFunction()
}
