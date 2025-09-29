import Foundation

final class TransformedPrimitive: Boundable, Intersectable, Sendable {

        init(
                //acceleratorIndex: AcceleratorIndex,
                accelerator: Accelerator,
                transform: Transform
        ) {
                //self.acceleratorIndex = acceleratorIndex
                self.accelerator = accelerator
                self.transform = transform
        }

        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitive? {
                let localRay = transform.inverse * ray
                return try accelerator.intersect_lean(
                        ray: localRay,
                        tHit: &tHit)
        }

        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction
        //) throws {
        //        let localRay = transform.inverse * ray
        //        // TODO: transform tHit?
        //        try accelerator.intersect(
        //                ray: localRay,
        //                tHit: &tHit,
        //                interaction: &interaction)
        //        if !interaction.valid {
        //                return
        //        }
        //        interaction = transform * interaction
        //}

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX) throws -> SurfaceInteraction
        {
                unimplemented()
        }

        @MainActor
        func worldBound() -> Bounds3f {
                let bound = transform * accelerator.worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        //let acceleratorIndex: AcceleratorIndex
        let accelerator: Accelerator
        let transform: Transform
}
