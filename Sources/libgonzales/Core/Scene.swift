struct Scene {

        init(
                lights: [Light],
                materials: [Material],
                meshes: TriangleMeshes,
                geometricPrimitives: [GeometricPrimitive],
                areaLights: [AreaLight],
                transformedPrimitives: [TransformedPrimitive],
                arena: TextureArena
        ) {
                self.lights = lights
                self.infiniteLights = lights.compactMap {
                        switch $0 {
                        case .infinite(let infiniteLight):
                                return infiniteLight
                        default:
                                return nil
                        }
                }
                self.materials = materials
                self.meshes = meshes
                self.geometricPrimitives = geometricPrimitives
                self.areaLights = areaLights
                self.transformedPrimitives = transformedPrimitives
                self.arena = arena
        }

        func intersect(primId: PrimId, ray: Ray, tHit: inout Real) throws -> Bool {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return try triangle.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                case .transformedPrimitive:
                        let transformedPrimitive = transformedPrimitives[primId.id1]
                        return try transformedPrimitive.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                case .areaLight:
                        let areaLight = areaLights[primId.id1]
                        return try areaLight.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                }
        }

        func getIntersectionData(
                primId: PrimId,
                ray: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) throws
                -> Bool {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return try triangle.getIntersectionData(
                                scene: self, ray: ray, tHit: &tHit, data: &data)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.getIntersectionData(
                                scene: self, ray: ray, tHit: &tHit, data: &data)
                case .transformedPrimitive:
                        let transformedPrimitive = transformedPrimitives[primId.id1]
                        return try transformedPrimitive.getIntersectionData(
                                scene: self, ray: ray, tHit: &tHit, data: &data)
                case .areaLight:
                        let areaLight = areaLights[primId.id1]
                        return try areaLight.getIntersectionData(
                                scene: self, ray: ray, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                primId: PrimId,
                data: TriangleIntersection,
                worldRay: Ray
        ) throws -> SurfaceInteraction? {
                switch primId.type {
                case .triangle:
                        let triangle = try Triangle(
                                meshIndex: primId.id1, number: primId.id2,
                                triangleMeshes: meshes)
                        return triangle.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay)
                case .geometricPrimitive:
                        let geometricPrimitive = geometricPrimitives[primId.id1]
                        return try geometricPrimitive.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay)
                case .transformedPrimitive:
                        let transformedPrimitive = transformedPrimitives[primId.id1]
                        return try transformedPrimitive.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay)
                case .areaLight:
                        let areaLight = areaLights[primId.id1]
                        return try areaLight.computeSurfaceInteraction(
                                scene: self, data: data, worldRay: worldRay)
                }
        }

        var lights: [Light]
        var infiniteLights: [InfiniteLight]
        var materials: [Material]
        var meshes: TriangleMeshes
        var geometricPrimitives: [GeometricPrimitive]
        var areaLights: [AreaLight]
        var transformedPrimitives: [TransformedPrimitive]
        var arena: TextureArena
}
