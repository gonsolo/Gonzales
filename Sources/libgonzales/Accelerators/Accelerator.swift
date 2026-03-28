import mojoKernel

struct Accelerator: Boundable, Intersectable, Sendable {

        init() {
                self.boundingHierarchy = BoundingHierarchy(primitives: [], bvh2Nodes: [])
                self.useGPU = false
        }

        init(boundingHierarchy: BoundingHierarchy, useGPU: Bool = false) {
                self.boundingHierarchy = boundingHierarchy
                self.useGPU = useGPU
        }

        private var boundingHierarchy: BoundingHierarchy
        let useGPU: Bool

        func uploadToGPU(scene: Scene) {
                boundingHierarchy.uploadToGPU(scene: scene)
        }

        // --- Closest Hit Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if useGPU {
                        return intersectGPU(scene: scene, ray: ray, tHit: &tHit)
                }
                return boundingHierarchy.intersect(
                        scene: scene,
                        ray: ray,
                        tHit: &tHit)
        }

        func intersectBatch(
                scene: Scene,
                rays: [Ray],
                tHits: inout [Real]
        ) -> [SurfaceInteraction?] {
                if useGPU {
                        return intersectBatchGPU(scene: scene, rays: rays, tHits: &tHits)
                }
                var results = [SurfaceInteraction?]()
                results.reserveCapacity(rays.count)
                for i in 0..<rays.count {
                        results.append(boundingHierarchy.intersect(scene: scene, ray: rays[i], tHit: &tHits[i]))
                }
                return results
        }

        // GPU closest-hit: same result interpretation as BoundingHierarchy.intersect
        private func intersectGPU(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                guard let result = boundingHierarchy.intersectGPU(
                        scene: scene, ray: ray, tHit: &tHit)
                else {
                        return nil
                }

                let id1 = Int(result.primId.id1)
                let rawId2 = Int(result.primId.id2)

                let type: PrimType
                if result.primId.type == 0 {
                        type = .triangle
                } else if result.primId.type == 1 {
                        type = .geometricPrimitive
                } else {
                        type = .areaLight
                }

                let meshIdx: Int
                let triIdx: Int
                if type == .geometricPrimitive || type == .areaLight {
                        meshIdx = rawId2 >> 32
                        triIdx = rawId2 & 0xFFFF_FFFF
                } else {
                        meshIdx = id1
                        triIdx = rawId2
                }

                let data = TriangleIntersection(
                        primId: PrimId(id1: meshIdx, id2: triIdx, type: .triangle),
                        tValue: Real(result.tHit),
                        barycentric0: Real(1.0 - result.u - result.v),
                        barycentric1: Real(result.u),
                        barycentric2: Real(result.v)
                )
                tHit = Real(result.tHit)
                return scene.computeSurfaceInteraction(
                        primId: PrimId(id1: id1, id2: rawId2, type: type), data: data, worldRay: ray)
        }

        // GPU batch closest-hit
        private func intersectBatchGPU(
                scene: Scene,
                rays: [Ray],
                tHits: inout [Real]
        ) -> [SurfaceInteraction?] {
                guard let results = boundingHierarchy.intersectBatchGPU(
                        scene: scene, rays: rays, tHits: &tHits)
                else {
                        return Array(repeating: nil, count: rays.count)
                }

                var interactions = [SurfaceInteraction?]()
                interactions.reserveCapacity(rays.count)

                for i in 0..<rays.count {
                        let result = results[i]
                        if result.hit == 0 {
                                interactions.append(nil)
                                continue
                        }

                        let id1 = Int(result.primId.id1)
                        let rawId2 = Int(result.primId.id2)

                        let type: PrimType
                        if result.primId.type == 0 {
                                type = .triangle
                        } else if result.primId.type == 1 {
                                type = .geometricPrimitive
                        } else {
                                type = .areaLight
                        }

                        let meshIdx: Int
                        let triIdx: Int
                        if type == .geometricPrimitive || type == .areaLight {
                                meshIdx = rawId2 >> 32
                                triIdx = rawId2 & 0xFFFF_FFFF
                        } else {
                                meshIdx = id1
                                triIdx = rawId2
                        }

                        let data = TriangleIntersection(
                                primId: PrimId(id1: meshIdx, id2: triIdx, type: .triangle),
                                tValue: Real(result.tHit),
                                barycentric0: Real(1.0 - result.u - result.v),
                                barycentric1: Real(result.u),
                                barycentric2: Real(result.v)
                        )
                        tHits[i] = Real(result.tHit)
                        let interaction = scene.computeSurfaceInteraction(
                                primId: PrimId(id1: id1, id2: rawId2, type: type), data: data, worldRay: rays[i])
                        interactions.append(interaction)
                }
                return interactions
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.objectBound(scene: scene)
        }

        func worldBound(scene: Scene) -> Bounds3f {
                return boundingHierarchy.worldBound(scene: scene)
        }
}
