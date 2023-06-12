protocol BsdfGeometryProtocol {

        var bsdfGeometry: BsdfGeometry { get }
}

extension BsdfGeometryProtocol {

        func worldToLocal(world: Vector) -> Vector {
                return bsdfGeometry.frame.worldToLocal(world: world)
        }

        func localToWorld(local: Vector) -> Vector {
                return bsdfGeometry.frame.localToWorld(local: local)
        }

        func isReflecting(wi: Vector, wo: Vector) -> Bool {
                return bsdfGeometry.isReflecting(wi: wi, wo: wo)
        }
}
