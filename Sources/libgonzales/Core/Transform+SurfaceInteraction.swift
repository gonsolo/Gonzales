extension Transform {

        static func * (left: Transform, right: SurfaceInteraction) -> SurfaceInteraction {
                return SurfaceInteraction(
                        valid: right.valid,
                        position: left * right.position,
                        normal: normalized(left * right.normal),
                        shadingNormal: normalized(left * right.shadingNormal),
                        outgoing: normalized(left * right.outgoing),
                        dpdu: left * right.dpdu,
                        uv: right.uv,
                        faceIndex: right.faceIndex,
                        materialIndex: right.materialIndex)
        }
}
