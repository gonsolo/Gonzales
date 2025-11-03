struct Scene {

        init() {
                lights = []
                infiniteLights = []
                materials = []
                meshes = TriangleMeshes(meshes: [])
                geometricPrimitives = []
        }

        mutating func addMeshes(meshes: TriangleMeshes) {
                self.meshes = meshes
        }

        mutating func addGeometricPrimivites(geometricPrimitives: [GeometricPrimitive]) {
                self.geometricPrimitives = geometricPrimitives
        }

        @MainActor
        mutating func addLights(lights: [Light]) {
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
        mutating func addMaterials(materials: [Material]) {
                self.materials = materials
        }

        func intersect(primId: PrimId, ray: Ray, tHit: inout FloatX) throws -> Bool {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return try triangle.intersect(scene: self, ray: ray, tHit: &tHit)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.intersect(scene: self, ray: ray, tHit: &tHit)
                case .transformedPrimitive:
                        let transformedPrimitive = transformedPrimitives[primId.id1]
                        return try transformedPrimitive.intersect(scene: self, ray: ray, tHit: &tHit)
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return try areaLight.intersect(scene: self, ray: ray, tHit: &tHit)
                }
        }

        func getIntersectionData(
                scene: Scene,
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
                        return try triangle.getIntersectionData(scene: scene, ray: ray, tHit: &tHit, data: &data)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.getIntersectionData(scene: scene, ray: ray, tHit: &tHit, data: &data)
                case .transformedPrimitive:
                        unimplemented()
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return try areaLight.getIntersectionData(scene: scene, ray: ray, tHit: &tHit, data: &data)
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
                                scene: self, data: data!, worldRay: worldRay, interaction: &interaction)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return geometricPrimitive.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay, interaction: &interaction)
                case .transformedPrimitive:
                        unimplemented()
                case .areaLight:
                        let areaLight = globalAreaLights[primId.id1]
                        return areaLight.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay, interaction: &interaction)
                }
        }

        var lights: [Light]
        var infiniteLights: [InfiniteLight]
        var materials: [Material]
        var meshes: TriangleMeshes
        var geometricPrimitives: [GeometricPrimitive]
}

