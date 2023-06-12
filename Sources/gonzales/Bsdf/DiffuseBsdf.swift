struct DiffuseBsdf: GlobalBsdf, LocalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func albedoLocal() -> RGBSpectrum { return reflectance }

        var reflectance: RGBSpectrum = white

        func worldToLocal(world: Vector) -> Vector {
                return bsdfGeometry.frame.worldToLocal(world: world)
        }

        func localToWorld(local: Vector) -> Vector {
                return bsdfGeometry.frame.localToWorld(local: local)
        }

        func isReflecting(wi: Vector, wo: Vector) -> Bool {
                return bsdfGeometry.isReflecting(wi: wi, wo: wo)
        }

        let bsdfGeometry: BsdfGeometry
}

struct DummyBsdf: GlobalBsdf, LocalBsdf {

        func worldToLocal(world: Vector) -> Vector {
                return nullVector
        }

        func localToWorld(local: Vector) -> Vector {
                return nullVector
        }

        func albedoLocal() -> RGBSpectrum { return white }

        func isReflecting(wi: Vector, wo: Vector) -> Bool { return false }

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum { return white }
}
