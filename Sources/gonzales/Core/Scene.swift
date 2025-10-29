public final class Scene {

        init() {
                accelerator = nil
                lights = []
                infiniteLights = []
                materials = []
                meshes = TriangleMeshes(meshes: [])
        }

        func addMeshes(meshes: TriangleMeshes) {
                self.meshes = meshes
        }

        @MainActor
        func addLights(lights: [Light]) {
                self.lights = lights
                self.infiniteLights = lights.compactMap {
                        switch $0 {
                        case .infinite(let infiniteLight):
                                return infiniteLight
                        default:
                                return nil
                        }
                }
        }

        @MainActor
        func addMaterials(materials: [Material]) {
                self.materials = materials
        }

        func addAccelerator(accelerator: Accelerator) {
                self.accelerator = accelerator
        }

        func intersect(primId: PrimId, ray: Ray, tHit: inout FloatX) throws -> Bool {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return try triangle.intersect(ray: ray, tHit: &tHit)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.intersect(ray: ray, tHit: &tHit)
                case .transformedPrimitive:
                        let transformedPrimitive = transformedPrimitives[primId.id1]
                        return try transformedPrimitive.intersect(ray: ray, tHit: &tHit)
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return try areaLight.intersect(ray: ray, tHit: &tHit)
                }
        }

        func getIntersectionData(
                primId: PrimId,
                ray: Ray,
                tHit: inout FloatX,
                data: inout TriangleIntersection) throws
                -> Bool
        {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return try triangle.getIntersectionData(ray: ray, tHit: &tHit, data: &data)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.getIntersectionData(ray: ray, tHit: &tHit, data: &data)
                case .transformedPrimitive:
                        unimplemented()
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return try areaLight.getIntersectionData(ray: ray, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                primId: PrimId,
                data: TriangleIntersection?,
                worldRay: Ray,
                interaction: inout SurfaceInteraction
        ) throws {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        triangle.computeSurfaceInteraction(
                                data: data!, worldRay: worldRay, interaction: &interaction)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return geometricPrimitive.computeSurfaceInteraction(
                                data: data, worldRay: worldRay, interaction: &interaction)
                case .transformedPrimitive:
                        unimplemented()
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return areaLight.computeSurfaceInteraction(
                                data: data, worldRay: worldRay, interaction: &interaction)
                }
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX
        ) throws -> Bool {
                return try accelerator.intersect(
                        ray: ray,
                        tHit: &tHit)
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction,
        ) throws {
                try accelerator.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        @MainActor
        func bound() -> Bounds3f {
                return accelerator.worldBound()
        }

        @MainActor
        func diameter() -> FloatX {
                return length(bound().diagonal())
        }

        var accelerator: Accelerator!
        var lights: [Light]
        var infiniteLights: [InfiniteLight]
        var materials: [Material]
        var meshes: TriangleMeshes
}

