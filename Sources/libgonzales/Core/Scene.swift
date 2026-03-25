import DevirtualizeMacro

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

        func intersect(primId: PrimId, ray: Ray, tHit: inout Real) -> Bool {
                return #dispatchPrimitive(id: primId, scene: self) { (p: Triangle) in
                        p.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                }
        }

        func getIntersectionData(
                primId: PrimId,
                ray: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        )
                -> Bool {
                return #dispatchPrimitive(id: primId, scene: self) { (p: Triangle) in
                        p.getIntersectionData(scene: self, ray: ray, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                primId: PrimId,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                return #dispatchPrimitive(id: primId, scene: self) { (p: Triangle) in
                        p.computeSurfaceInteraction(scene: self, data: data, worldRay: worldRay)
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
