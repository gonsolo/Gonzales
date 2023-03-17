struct PhaseFunction {

        func samplePhase(wo: Vector, sampler: Sampler) -> Vector {
                // TODO
                return Vector()
        }
}

struct MediumInteraction: Interaction {

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
                // TODO
                return (Ray(), 0)
        }

        func spawnRay(inDirection direction: Vector) -> Ray {
                // TODO
                return (Ray())
        }

        var dpdu = Vector()
        var faceIndex = 0
        var normal = Normal()
        var position = Point()
        var shadingNormal = Normal()
        var uv = Point2F()
        var wo = Vector()

        var phase = PhaseFunction()
}
