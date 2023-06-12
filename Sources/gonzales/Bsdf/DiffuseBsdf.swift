struct DiffuseBsdf: GlobalBsdf {

        func evaluateLocal(wo: Vector, wi: Vector) -> RGBSpectrum {
                return reflectance / FloatX.pi
        }

        func albedo() -> RGBSpectrum { return reflectance }

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
