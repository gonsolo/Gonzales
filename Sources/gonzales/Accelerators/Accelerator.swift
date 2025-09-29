enum Accelerator: Boundable, Intersectable, Sendable {

        case boundingHierarchy(BoundingHierarchy)
        //case embree(EmbreeAccelerator)
        //case optix(Optix)

        func intersect(
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        for i in 0..<rays.count {
                                if !skips[i] {
                                        //try boundingHierarchy.intersect(
                                        //        ray: rays[i],
                                        //        tHit: &tHits[i],
                                        //        interaction: &interactions[i])
                                        let candidate = try boundingHierarchy.intersect_lean(ray: rays[i], tHit: &tHits[i])
                                        if candidate != nil {
                                                let interaction = try boundingHierarchy.computeInteraction(
                                                        ray: rays[i], tHit: &tHits[i])
                                                interactions[i] = interaction
                                        }
                                }
                        }
                //case .embree(let embree):
                //        for i in 0..<rays.count {
                //                if !skips[i] {
                //                        try embree.intersect(
                //                                ray: rays[i],
                //                                tHit: &tHits[i],
                //                                interaction: &interactions[i])
                //                }
                //        }
                //case .optix(let optix):
                //        try optix.intersect(
                //                rays: rays,
                //                tHits: &tHits,
                //                interactions: &interactions,
                //                skips: skips)
                }
        }

        func intersect_lean(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> IntersectablePrimitive? {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return try boundingHierarchy.intersect_lean(
                                ray: ray,
                                tHit: &tHit)
                }
        }

        //func intersect(
        //        ray: Ray,
        //        tHit: inout FloatX,
        //        interaction: inout SurfaceInteraction
        //) throws {
        //        switch self {
        //        case .boundingHierarchy(let boundingHierarchy):
        //                try boundingHierarchy.intersect(
        //                        ray: ray,
        //                        tHit: &tHit,
        //                        interaction: &interaction)
        //        //case .embree(let embree):
        //        //        try embree.intersect(
        //        //                ray: ray,
        //        //                tHit: &tHit,
        //        //                interaction: &interaction)
        //        //case .optix(let optix):
        //        //        try optix.intersect(
        //        //                ray: ray,
        //        //                tHit: &tHit,
        //        //                interaction: &interaction)
        //        }
        //}

        func computeInteraction(
                ray: Ray,
                tHit: inout FloatX) throws -> SurfaceInteraction
        {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return try boundingHierarchy.computeInteraction(
                                ray: ray,
                                tHit: &tHit)
                }
        }

        @MainActor
        func objectBound() -> Bounds3f {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.objectBound()
                //case .embree(let embree):
                //        return embree.objectBound()
                //case .optix(let optix):
                //        return optix.objectBound()
                }
        }

        @MainActor
        func worldBound() -> Bounds3f {
                switch self {
                case .boundingHierarchy(let boundingHierarchy):
                        return boundingHierarchy.worldBound()
                //case .embree(let embree):
                //        return embree.worldBound()
                //case .optix(let optix):
                //        return optix.worldBound()
                }
        }

}
