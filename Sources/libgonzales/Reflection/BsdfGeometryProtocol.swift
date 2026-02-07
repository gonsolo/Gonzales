public protocol BsdfFrameProtocol {

        var bsdfFrame: BsdfFrame { get }
}

extension BsdfFrameProtocol {

        func worldToLocal(world: Vector) -> Vector {
                return bsdfFrame.shadingFrame.worldToLocal(world: world)
        }

        func localToWorld(local: Vector) -> Vector {
                return bsdfFrame.shadingFrame.localToWorld(local: local)
        }

        func isReflecting(incident: Vector, outgoing: Vector) -> Bool {
                return bsdfFrame.isReflecting(incident: incident, outgoing: outgoing)
        }
}
