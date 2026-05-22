import mojoKernel

final class BoundingHierarchy: Boundable, Intersectable, @unchecked Sendable {

        let bvh2NodesPointer: UnsafeMutablePointer<BVH2Node>
        let bvh2NodesCount: Int
        let primIdsPointer: UnsafeMutablePointer<PrimId>
        let primIdsCount: Int
        var gpuSceneHandle: UnsafeMutableRawPointer?
        var materialsC: [Material_C] = []
        var areaLightsC: [AreaLight_C] = []

        init(primitives: [IntersectablePrimitive], bvh2Nodes: [BVH2Node]) {
                self.bvh2NodesCount = bvh2Nodes.count
                self.bvh2NodesPointer = UnsafeMutablePointer<BVH2Node>.allocate(
                        capacity: max(bvh2Nodes.count, 1))
                if !bvh2Nodes.isEmpty {
                        self.bvh2NodesPointer.initialize(from: bvh2Nodes, count: bvh2Nodes.count)
                }
                self.gpuSceneHandle = nil

                var ids = [PrimId]()
                for primitive in primitives {
                        switch primitive {
                        case .geometricPrimitive(let geometricPrimitive):
                                if case .triangle(let triangle) = geometricPrimitive.shape {
                                        // Pack meshIndex high, triIndex low into id2 so Mojo can intersect
                                        let packed = (triangle.meshIndex << 32) | (triangle.triangleIndex / 3)
                                        let primId = PrimId(
                                                id1: geometricPrimitive.idx, id2: packed,
                                                type: .geometricPrimitive,
                                                materialIndex: geometricPrimitive.materialIndex)
                                        ids.append(primId)
                                } else {
                                        let primId = PrimId(
                                                id1: geometricPrimitive.idx, id2: -1,
                                                type: .geometricPrimitive,
                                                materialIndex: geometricPrimitive.materialIndex)
                                        ids.append(primId)
                                }
                        case .triangle(let triangle):
                                let primId = PrimId(
                                        id1: triangle.meshIndex, id2: triangle.triangleIndex, type: .triangle)
                                ids.append(primId)
                        case .transformedPrimitive(let transformedPrimitive):
                                let primId = PrimId(
                                        id1: transformedPrimitive.idx, id2: -1, type: .transformedPrimitive)
                                ids.append(primId)
                        case .areaLight(let areaLight):
                                if case .triangle(let triangle) = areaLight.shape {
                                        let packed = (triangle.meshIndex << 32) | (triangle.triangleIndex / 3)
                                        let primId = PrimId(id1: areaLight.idx, id2: packed, type: .areaLight)
                                        ids.append(primId)
                                } else {
                                        let primId = PrimId(id1: areaLight.idx, id2: -1, type: .areaLight)
                                        ids.append(primId)
                                }
                        }
                }
                self.primIdsCount = ids.count
                self.primIdsPointer = UnsafeMutablePointer<PrimId>.allocate(capacity: max(ids.count, 1))
                if !ids.isEmpty {
                        self.primIdsPointer.initialize(from: ids, count: ids.count)
                }

                print(
                        "LAYOUT: BVH2Node     stride=\(MemoryLayout<BVH2Node>.stride) "
                                + "size=\(MemoryLayout<BVH2Node>.size) "
                                + "align=\(MemoryLayout<BVH2Node>.alignment)")
                print("LAYOUT: BVH2 nodes=\(bvh2Nodes.count) prims=\(ids.count)")
                print("LAYOUT: BVH2 memory=\(bvh2Nodes.count * MemoryLayout<BVH2Node>.stride) bytes")
        }

        deinit {
                if let handle = gpuSceneHandle {
                        mojo_gpu_free_scene(handle)
                        gpuSceneHandle = nil
                }
                if bvh2NodesCount > 0 {
                        bvh2NodesPointer.deinitialize(count: bvh2NodesCount)
                }
                bvh2NodesPointer.deallocate()
                if primIdsCount > 0 {
                        primIdsPointer.deinitialize(count: primIdsCount)
                }
                primIdsPointer.deallocate()
        }
}

extension BoundingHierarchy {

        func prepareMaterials(scene: Scene) {
                guard materialsC.isEmpty else { return }

                var mats = [Material_C]()
                let dummyInteraction = SurfaceInteraction()
                var matTypeCounts = [String: Int]()
                for material in scene.materials {
                        // Default to diffuse gray so paths always bounce
                        var cMat = Material_C(type: 1, albedo: (0.5, 0.5, 0.5), emission: (0, 0, 0))
                        switch material {
                        case .diffuse(let diffuse):
                                matTypeCounts["diffuse", default: 0] += 1
                                let evaluation = diffuse.reflectance.evaluate(
                                        at: dummyInteraction, arena: scene.arena)
                                var rgb = RgbSpectrum(intensity: 0.0)
                                if let float = evaluation as? Real {
                                        rgb = RgbSpectrum(intensity: float)
                                } else if let spec = evaluation as? RgbSpectrum {
                                        rgb = spec
                                }
                                cMat.albedo = (Float(rgb.red), Float(rgb.green), Float(rgb.blue))
                        case .coatedDiffuse(let coatedDiffuse):
                                matTypeCounts["coatedDiffuse", default: 0] += 1
                                var ior: Float = 1.5
                                if let f = coatedDiffuse.refractiveIndex.evaluate(
                                        at: dummyInteraction, arena: scene.arena) as? Real
                                {
                                        ior = Float(f)
                                }
                                let refl = coatedDiffuse.reflectance.evaluate(
                                        at: dummyInteraction, arena: scene.arena)
                                if let rgb = refl as? RgbSpectrum {
                                        cMat.albedo = (Float(rgb.red), Float(rgb.green), Float(rgb.blue))
                                } else if let f = refl as? Real {
                                        cMat.albedo = (Float(f), Float(f), Float(f))
                                }
                                cMat.type = 5
                                cMat.emission = (ior, 0, 0)
                        case .conductor:
                                matTypeCounts["conductor", default: 0] += 1
                                cMat.type = 3
                                cMat.albedo = (0.7, 0.7, 0.7)
                        case .coatedConductor:
                                matTypeCounts["coatedConductor", default: 0] += 1
                                cMat.type = 3
                                cMat.albedo = (0.7, 0.7, 0.7)
                        case .dielectric(let dielectric):
                                matTypeCounts["dielectric", default: 0] += 1
                                var ior: Float = 1.5
                                if let f = dielectric.refractiveIndex.evaluate(
                                        at: dummyInteraction, arena: scene.arena) as? Real
                                {
                                        ior = Float(f)
                                }
                                cMat.type = 4
                                cMat.albedo = (ior, 0, 0)
                        case .diffuseTransmission(let dt):
                                matTypeCounts["diffuseTransmission", default: 0] += 1
                                var reflR: Float = 1; var reflG: Float = 1; var reflB: Float = 1
                                var transR: Float = 1; var transG: Float = 1; var transB: Float = 1
                                var scale: Float = 1
                                if let rgb = dt.reflectance.evaluate(at: dummyInteraction, arena: scene.arena) as? RgbSpectrum {
                                        reflR = Float(rgb.red); reflG = Float(rgb.green); reflB = Float(rgb.blue)
                                } else if let f = dt.reflectance.evaluate(at: dummyInteraction, arena: scene.arena) as? Real {
                                        reflR = Float(f); reflG = Float(f); reflB = Float(f)
                                }
                                if let rgb = dt.transmittance.evaluate(at: dummyInteraction, arena: scene.arena) as? RgbSpectrum {
                                        transR = Float(rgb.red); transG = Float(rgb.green); transB = Float(rgb.blue)
                                } else if let f = dt.transmittance.evaluate(at: dummyInteraction, arena: scene.arena) as? Real {
                                        transR = Float(f); transG = Float(f); transB = Float(f)
                                }
                                if let f = dt.scale.evaluate(at: dummyInteraction, arena: scene.arena) as? Real {
                                        scale = Float(f)
                                }
                                cMat.type = 6
                                cMat.albedo = (reflR * scale, reflG * scale, reflB * scale)
                                cMat.emission = (transR * scale, transG * scale, transB * scale)
                        default:
                                matTypeCounts["other", default: 0] += 1
                        }
                        mats.append(cMat)
                }
                print("Shading: Material distribution: \(matTypeCounts)")
                for areaLight in scene.areaLights {
                        var cMat = Material_C(type: 2, albedo: (0, 0, 0), emission: (0, 0, 0))
                        let rgb = areaLight.brightness
                        cMat.emission = (Float(rgb.red), Float(rgb.green), Float(rgb.blue))
                        mats.append(cMat)
                }
                let baseMatIndex = mats.count - scene.areaLights.count
                for i in 0..<primIdsCount {
                        if primIdsPointer[i].type == .areaLight {
                                let lightIdx = primIdsPointer[i].id1
                                primIdsPointer[i] = PrimId(
                                        id1: lightIdx,
                                        id2: primIdsPointer[i].id2,
                                        type: .areaLight,
                                        materialIndex: baseMatIndex + lightIdx
                                )
                        }
                }
                self.materialsC = mats

                // Build emissive triangle list for Mojo NEE
                var alights = [AreaLight_C]()
                for areaLight in scene.areaLights {
                        if case .triangle(let triangle) = areaLight.shape {
                                let brightness = areaLight.brightness
                                alights.append(AreaLight_C(
                                        meshIdx: Int32(triangle.meshIndex),
                                        triBaseVidx: Int32(triangle.triangleIndex),
                                        emissionR: Float(brightness.red),
                                        emissionG: Float(brightness.green),
                                        emissionB: Float(brightness.blue),
                                        _pad: 0))
                        }
                }
                self.areaLightsC = alights
        }

        func uploadToGPU(scene: Scene) {
                guard bvh2NodesCount > 0 else {
                        print("GPU: No BVH nodes to upload")
                        return
                }

                prepareMaterials(scene: scene)

                // Prepare per-mesh size arrays for the Mojo upload function
                let meshCount = scene.meshesC.count
                var pointsCounts = [Int64]()
                var faceIndicesCounts = [Int64]()
                var vertexIndicesCounts = [Int64]()

                for mesh in scene.meshes.meshes {
                        // Points: each Point3 is SIMD4<Float32> = 4 floats
                        pointsCounts.append(Int64(mesh.pointCount * 4))
                        faceIndicesCounts.append(Int64(mesh.faceIndices.count))
                        // vertexIndices: numberTriangles * 3 entries
                        vertexIndicesCounts.append(Int64(mesh.numberTriangles * 3))
                }

                let handle = scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        pointsCounts.withUnsafeBufferPointer { ptsPtr in
                                faceIndicesCounts.withUnsafeBufferPointer { fiPtr in
                                        vertexIndicesCounts.withUnsafeBufferPointer { viPtr in
                                                self.materialsC.withUnsafeBufferPointer { matPtr in
                                                        mojo_gpu_upload_scene(
                                                                UnsafeRawPointer(bvh2NodesPointer)
                                                                        .assumingMemoryBound(
                                                                                to: mojoKernel.BVH2Node.self),
                                                                Int64(bvh2NodesCount),
                                                                UnsafeRawPointer(primIdsPointer)
                                                                        .assumingMemoryBound(
                                                                                to: PrimId_C.self),
                                                                Int64(primIdsCount),
                                                                meshesPtr.baseAddress,
                                                                Int64(meshCount),
                                                                ptsPtr.baseAddress,
                                                                fiPtr.baseAddress,
                                                                viPtr.baseAddress,
                                                                matPtr.baseAddress,
                                                                Int64(self.materialsC.count)
                                                        )
                                                }
                                        }
                                }
                        }
                }
                gpuSceneHandle = handle
        }

        // --- GPU Batch Intersect ---
        func intersectGPU(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Intersection_C? {
                guard let handle = gpuSceneHandle else { return nil }

                var rayC = Ray_C(
                        orgX: Float(ray.origin.x), orgY: Float(ray.origin.y),
                        orgZ: Float(ray.origin.z),
                        dirX: Float(ray.direction.x), dirY: Float(ray.direction.y),
                        dirZ: Float(ray.direction.z)
                )
                var tMaxValue = Float(tHit)
                var result = Intersection_C()

                withUnsafePointer(to: &rayC) { rayPtr in
                        withUnsafePointer(to: &tMaxValue) { tMaxPtr in
                                withUnsafeMutablePointer(to: &result) { resPtr in
                                        mojo_gpu_traverse_batch(
                                                handle,
                                                rayPtr,
                                                tMaxPtr,
                                                1,
                                                resPtr
                                        )
                                }
                        }
                }

                if result.hit != 0 {
                        tHit = Real(result.tHit)
                        return result
                }
                return nil
        }

        func intersectBatchGPU(
                scene: Scene,
                rays: [Ray],
                tHits: inout [Real]
        ) -> [Intersection_C]? {
                guard let handle = gpuSceneHandle else { return nil }

                // Map payloads to flat buffers
                let raysC = rays.map { ray in
                        Ray_C(
                                orgX: Float(ray.origin.x), orgY: Float(ray.origin.y),
                                orgZ: Float(ray.origin.z),
                                dirX: Float(ray.direction.x), dirY: Float(ray.direction.y),
                                dirZ: Float(ray.direction.z)
                        )
                }
                let tMaxValues = tHits.map { Float($0) }
                var results = [Intersection_C](repeating: Intersection_C(), count: rays.count)

                raysC.withUnsafeBufferPointer { raysPtr in
                        tMaxValues.withUnsafeBufferPointer { tMaxPtr in
                                results.withUnsafeMutableBufferPointer { resPtr in
                                        mojo_gpu_traverse_batch(
                                                handle,
                                                raysPtr.baseAddress!,
                                                tMaxPtr.baseAddress!,
                                                Int64(rays.count),
                                                resPtr.baseAddress!
                                        )
                                }
                        }
                }
                return results
        }

        // --- CPU Batch Intersect (parallelize via Mojo) ---
        func intersectBatchCPU(
                scene: Scene,
                rays: [Ray],
                tHits: inout [Real]
        ) -> [Intersection_C] {
                let raysC = rays.map { ray in
                        Ray_C(
                                orgX: Float(ray.origin.x), orgY: Float(ray.origin.y),
                                orgZ: Float(ray.origin.z),
                                dirX: Float(ray.direction.x), dirY: Float(ray.direction.y),
                                dirZ: Float(ray.direction.z)
                        )
                }
                let tMaxValues = tHits.map { Float($0) }
                var results = [Intersection_C](repeating: Intersection_C(), count: rays.count)

                scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        self.materialsC.withUnsafeBufferPointer { matPtr in
                                var desc = SceneDescriptor2_C(
                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer)
                                                .assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                        primIds: UnsafeRawPointer(primIdsPointer)
                                                .assumingMemoryBound(to: PrimId_C.self),
                                        meshes: meshesPtr.baseAddress!,
                                        meshCount: Int64(scene.meshesC.count),
                                        materials: matPtr.baseAddress,
                                        materialCount: Int64(self.materialsC.count),
                                        areaLights: nil,
                                        areaLightCount: 0
                                )

                                raysC.withUnsafeBufferPointer { raysPtr in
                                        tMaxValues.withUnsafeBufferPointer { tMaxPtr in
                                                results.withUnsafeMutableBufferPointer { resPtr in
                                                        withUnsafePointer(to: &desc) { descPtr in
                                                                mojo_cpu_traverse_batch(
                                                                        descPtr,
                                                                        raysPtr.baseAddress!,
                                                                        tMaxPtr.baseAddress!,
                                                                        Int64(rays.count),
                                                                        resPtr.baseAddress!
                                                                )
                                                        }
                                                }
                                        }
                                }
                        }
                }
                return results
        }

        // --- CPU Batch Shade (parallelize via Mojo) ---
        func cpuShadeBatch(
                scene: Scene,
                pathStatesC: inout [PathState_C],
                intersectionsC: [Intersection_C]
        ) {
                let count = Int64(pathStatesC.count)
                pathStatesC.withUnsafeMutableBufferPointer { pathsPtr in
                        intersectionsC.withUnsafeBufferPointer { interPtr in
                                scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                                        self.materialsC.withUnsafeBufferPointer { matPtr in
                                                mojo_cpu_shade_batch(
                                                        pathsPtr.baseAddress!,
                                                        count,
                                                        interPtr.baseAddress!,
                                                        meshesPtr.baseAddress!,
                                                        matPtr.baseAddress!
                                                )
                                        }
                                }
                        }
                }
        }

        // --- Occlusion Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> Bool {
                if bvh2NodesCount == 0 { return false }

                return scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        self.materialsC.withUnsafeBufferPointer { matPtr in
                                var desc = SceneDescriptor2_C(
                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer).assumingMemoryBound(
                                                to: mojoKernel.BVH2Node.self),
                                        primIds: UnsafeRawPointer(primIdsPointer).assumingMemoryBound(
                                                to: PrimId_C.self),
                                        meshes: meshesPtr.baseAddress,
                                        meshCount: Int64(scene.meshesC.count),
                                        materials: matPtr.baseAddress,
                                        materialCount: Int64(self.materialsC.count),
                                        areaLights: nil,
                                        areaLightCount: 0
                                )
                                var rayC = Ray_C(
                                        orgX: Float(ray.origin.x), orgY: Float(ray.origin.y),
                                        orgZ: Float(ray.origin.z),
                                        dirX: Float(ray.direction.x), dirY: Float(ray.direction.y),
                                        dirZ: Float(ray.direction.z)
                                )
                                var result = Intersection_C()
                                withUnsafePointer(to: &desc) { descP in
                                        withUnsafePointer(to: &rayC) { rayP in
                                                withUnsafeMutablePointer(to: &result) { resP in
                                                        mojo_traverse_bvh2(descP, rayP, Float(tHit), resP)
                                                }
                                        }
                                }
                                if result.hit != 0 {
                                        tHit = Real(result.tHit)
                                        return true
                                }
                                return false
                        }
                }
        }

        // --- Closest Hit Query ---
        func intersect(
                scene: Scene,
                ray: Ray,
                tHit: inout Real
        ) -> SurfaceInteraction? {
                if bvh2NodesCount == 0 { return nil }

                let result = scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                        self.materialsC.withUnsafeBufferPointer { matPtr in
                                var desc = SceneDescriptor2_C(
                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer).assumingMemoryBound(
                                                to: mojoKernel.BVH2Node.self),
                                        primIds: UnsafeRawPointer(primIdsPointer).assumingMemoryBound(
                                                to: PrimId_C.self),
                                        meshes: meshesPtr.baseAddress,
                                        meshCount: Int64(scene.meshesC.count),
                                        materials: matPtr.baseAddress,
                                        materialCount: Int64(self.materialsC.count),
                                        areaLights: nil,
                                        areaLightCount: 0
                                )
                                var rayC = Ray_C(
                                        orgX: Float(ray.origin.x), orgY: Float(ray.origin.y),
                                        orgZ: Float(ray.origin.z),
                                        dirX: Float(ray.direction.x), dirY: Float(ray.direction.y),
                                        dirZ: Float(ray.direction.z)
                                )
                                var result = Intersection_C()
                                withUnsafePointer(to: &desc) { descP in
                                        withUnsafePointer(to: &rayC) { rayP in
                                                withUnsafeMutablePointer(to: &result) { resP in
                                                        mojo_traverse_bvh2(descP, rayP, Float(tHit), resP)
                                                }
                                        }
                                }
                                return result
                        }
                }

                if result.hit != 0 {
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

                return nil
        }

        // --- Camera ray generation + full path trace in Mojo ---
        func renderTile(
                rasterToCamera: [Float], cameraToWorld: [Float],
                samples: [PixelSample_C], scene: Scene,
                results: inout [TileResult_C], maxDepth: Int
        ) {
                let count = Int64(samples.count)
                samples.withUnsafeBufferPointer { samplesPtr in
                        scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                                self.materialsC.withUnsafeBufferPointer { matPtr in
                                        self.areaLightsC.withUnsafeBufferPointer { alPtr in
                                                var desc = SceneDescriptor2_C(
                                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer)
                                                                .assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                                        primIds: UnsafeRawPointer(primIdsPointer)
                                                                .assumingMemoryBound(to: PrimId_C.self),
                                                        meshes: meshesPtr.baseAddress!,
                                                        meshCount: Int64(scene.meshesC.count),
                                                        materials: matPtr.baseAddress,
                                                        materialCount: Int64(self.materialsC.count),
                                                        areaLights: alPtr.baseAddress,
                                                        areaLightCount: Int64(self.areaLightsC.count)
                                                )
                                                results.withUnsafeMutableBufferPointer { resPtr in
                                                        rasterToCamera.withUnsafeBufferPointer { rtcPtr in
                                                                cameraToWorld.withUnsafeBufferPointer { ctwPtr in
                                                                        withUnsafePointer(to: &desc) { descPtr in
                                                                                mojo_render_tile(
                                                                                        rtcPtr.baseAddress!,
                                                                                        ctwPtr.baseAddress!,
                                                                                        samplesPtr.baseAddress!,
                                                                                        count,
                                                                                        descPtr,
                                                                                        resPtr.baseAddress!,
                                                                                        Int32(maxDepth)
                                                                                )
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        func renderTileV2(
                rasterToCamera: [Float], cameraToWorld: [Float],
                bounds: Bounds2i,
                sobolSeed: Int32, log2SamplesPerPixel: Int32, nBase4Digits: Int32,
                samplesPerPixel: Int32,
                filterSigma: Float, filterSupportX: Float, filterSupportY: Float,
                filterNormX: Float, filterNormY: Float, filterWeight: Float,
                rngSeed: UInt64,
                scene: Scene, results: inout [TileResult_C], maxDepth: Int
        ) {
                sobolMatrices.withUnsafeBufferPointer { matPtr in
                        var sp = TileSamplerParams_C(
                                sobolMatrices: matPtr.baseAddress!,
                                rngSeed: rngSeed,
                                sobolSeed: sobolSeed,
                                log2SamplesPerPixel: log2SamplesPerPixel,
                                nBase4Digits: nBase4Digits,
                                samplesPerPixel: samplesPerPixel,
                                filterSigma: filterSigma,
                                filterSupportX: filterSupportX,
                                filterSupportY: filterSupportY,
                                filterNormX: filterNormX,
                                filterNormY: filterNormY,
                                filterWeight: filterWeight)
                        scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                                self.materialsC.withUnsafeBufferPointer { matPtr2 in
                                        self.areaLightsC.withUnsafeBufferPointer { alPtr in
                                                var desc = SceneDescriptor2_C(
                                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer)
                                                                .assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                                        primIds: UnsafeRawPointer(primIdsPointer)
                                                                .assumingMemoryBound(to: PrimId_C.self),
                                                        meshes: meshesPtr.baseAddress!,
                                                        meshCount: Int64(scene.meshesC.count),
                                                        materials: matPtr2.baseAddress,
                                                        materialCount: Int64(self.materialsC.count),
                                                        areaLights: alPtr.baseAddress,
                                                        areaLightCount: Int64(self.areaLightsC.count))
                                                results.withUnsafeMutableBufferPointer { resPtr in
                                                        rasterToCamera.withUnsafeBufferPointer { rtcPtr in
                                                                cameraToWorld.withUnsafeBufferPointer { ctwPtr in
                                                                        withUnsafePointer(to: &sp) { spPtr in
                                                                                withUnsafePointer(to: &desc) { descPtr in
                                                                                        mojo_render_tile_v2(
                                                                                                rtcPtr.baseAddress!,
                                                                                                ctwPtr.baseAddress!,
                                                                                                Int32(bounds.pMin.x), Int32(bounds.pMin.y),
                                                                                                Int32(bounds.pMax.x), Int32(bounds.pMax.y),
                                                                                                spPtr, descPtr,
                                                                                                resPtr.baseAddress!,
                                                                                                Int32(maxDepth))
                                                                                }
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        // --- All-tiles Mojo render (step 15) ---
        // Renders the full image in one call; Mojo handles tile iteration.
        func renderAllTilesMojo(
                rasterToCamera: [Float], cameraToWorld: [Float],
                bounds: Bounds2i, tileW: Int, tileH: Int,
                sobolSeed: Int32, log2SamplesPerPixel: Int32, nBase4Digits: Int32,
                samplesPerPixel: Int32,
                filterSigma: Float, filterSupportX: Float, filterSupportY: Float,
                filterNormX: Float, filterNormY: Float, filterWeight: Float,
                rngSeed: UInt64,
                scene: Scene, results: inout [TileResult_C], maxDepth: Int
        ) {
                sobolMatrices.withUnsafeBufferPointer { matPtr in
                        var sp = TileSamplerParams_C(
                                sobolMatrices: matPtr.baseAddress!,
                                rngSeed: rngSeed,
                                sobolSeed: sobolSeed,
                                log2SamplesPerPixel: log2SamplesPerPixel,
                                nBase4Digits: nBase4Digits,
                                samplesPerPixel: samplesPerPixel,
                                filterSigma: filterSigma,
                                filterSupportX: filterSupportX,
                                filterSupportY: filterSupportY,
                                filterNormX: filterNormX,
                                filterNormY: filterNormY,
                                filterWeight: filterWeight)
                        scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                                materialsC.withUnsafeBufferPointer { matPtr2 in
                                        areaLightsC.withUnsafeBufferPointer { alPtr in
                                                var desc = SceneDescriptor2_C(
                                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer)
                                                                .assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                                        primIds: UnsafeRawPointer(primIdsPointer)
                                                                .assumingMemoryBound(to: PrimId_C.self),
                                                        meshes: meshesPtr.baseAddress!,
                                                        meshCount: Int64(scene.meshesC.count),
                                                        materials: matPtr2.baseAddress,
                                                        materialCount: Int64(materialsC.count),
                                                        areaLights: alPtr.baseAddress,
                                                        areaLightCount: Int64(areaLightsC.count))
                                                results.withUnsafeMutableBufferPointer { resPtr in
                                                        rasterToCamera.withUnsafeBufferPointer { rtcPtr in
                                                                cameraToWorld.withUnsafeBufferPointer { ctwPtr in
                                                                        withUnsafePointer(to: &sp) { spPtr in
                                                                                withUnsafePointer(to: &desc) { descPtr in
                                                                                        mojo_render_all_tiles(
                                                                                                rtcPtr.baseAddress!,
                                                                                                ctwPtr.baseAddress!,
                                                                                                Int32(bounds.pMin.x),
                                                                                                Int32(bounds.pMin.y),
                                                                                                Int32(bounds.pMax.x),
                                                                                                Int32(bounds.pMax.y),
                                                                                                Int32(tileW), Int32(tileH),
                                                                                                spPtr, descPtr,
                                                                                                resPtr.baseAddress!,
                                                                                                Int32(maxDepth))
                                                                                }
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

        // --- Full multi-bounce CPU render (all bounces in Mojo) ---
        func renderPaths(scene: Scene, pathStatesC: inout [PathState_C], maxDepth: Int) {
                let count = Int64(pathStatesC.count)
                pathStatesC.withUnsafeMutableBufferPointer { pathsPtr in
                        scene.meshesC.withUnsafeBufferPointer { meshesPtr in
                                self.materialsC.withUnsafeBufferPointer { matPtr in
                                        self.areaLightsC.withUnsafeBufferPointer { alPtr in
                                                var desc = SceneDescriptor2_C(
                                                        bvh2Nodes: UnsafeRawPointer(bvh2NodesPointer)
                                                                .assumingMemoryBound(to: mojoKernel.BVH2Node.self),
                                                        primIds: UnsafeRawPointer(primIdsPointer)
                                                                .assumingMemoryBound(to: PrimId_C.self),
                                                        meshes: meshesPtr.baseAddress!,
                                                        meshCount: Int64(scene.meshesC.count),
                                                        materials: matPtr.baseAddress,
                                                        materialCount: Int64(self.materialsC.count),
                                                        areaLights: alPtr.baseAddress,
                                                        areaLightCount: Int64(self.areaLightsC.count)
                                                )
                                                withUnsafePointer(to: &desc) { descPtr in
                                                        mojo_render_paths(
                                                                descPtr,
                                                                pathsPtr.baseAddress!,
                                                                count,
                                                                Int32(maxDepth)
                                                        )
                                                }
                                        }
                                }
                        }
                }
        }

        func objectBound(scene: Scene) -> Bounds3f {
                return worldBound(scene: scene)
        }

        func worldBound(scene _: Scene) -> Bounds3f {
                if bvh2NodesCount == 0 {
                        return Bounds3f()
                } else {
                        let root = bvh2NodesPointer[0]
                        return Bounds3f(
                                first: Point3(
                                        x: Real(root.boundsMinX), y: Real(root.boundsMinY),
                                        z: Real(root.boundsMinZ)),
                                second: Point3(
                                        x: Real(root.boundsMaxX), y: Real(root.boundsMaxY),
                                        z: Real(root.boundsMaxZ))
                        )
                }
        }
}

enum PrimType: UInt8 {
        case triangle
        case geometricPrimitive
        case transformedPrimitive
        case areaLight
}

struct PrimId {
        init() {
                id1 = 0
                id2 = 0
                materialIndex = 0
                type = .triangle
        }
        init(id1: Int, id2: Int, type: PrimType, materialIndex: Int = 0) {
                self.id1 = id1
                self.id2 = id2
                self.materialIndex = materialIndex
                self.type = type
        }

        let id1: Int
        let id2: Int
        let materialIndex: Int
        let type: PrimType
}
