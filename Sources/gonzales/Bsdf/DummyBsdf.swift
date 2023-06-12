struct DummyBsdf: GlobalBsdf, LocalBsdf {

        func worldToLocal(world: Vector) -> Vector {
                return nullVector
        }

        func localToWorld(local: Vector) -> Vector {
                return nullVector
        }

        func albedo() -> RGBSpectrum { return white }

        func isReflecting(wi: Vector, wo: Vector) -> Bool { return false }

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum { return white }
}
