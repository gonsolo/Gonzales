struct DummyBsdf: GlobalBsdf {

        func worldToLocal(world _: Vector) -> Vector {
                return nullVector
        }

        func localToWorld(local _: Vector) -> Vector {
                return nullVector
        }

        func albedo() -> RgbSpectrum { return white }

        func isReflecting(incident _: Vector, outgoing _: Vector) -> Bool { return false }

        func evaluateLocal(outgoing _: Vector, incident _: Vector) -> RgbSpectrum { return white }

        var bsdfFrame: BsdfFrame { return BsdfFrame() }
}
