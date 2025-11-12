struct DummyBsdf: GlobalBsdf {

        func worldToLocal(world _: Vector) -> Vector {
                return nullVector
        }

        func localToWorld(local _: Vector) -> Vector {
                return nullVector
        }

        func albedo() -> RgbSpectrum { return white }

        func isReflecting(wi _: Vector, wo _: Vector) -> Bool { return false }

        func evaluateLocal(wo _: Vector, wi _: Vector) -> RgbSpectrum { return white }

        var bsdfFrame: BsdfFrame { return BsdfFrame() }
}
