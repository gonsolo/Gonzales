import DevirtualizeMacro
import mojoKernel

final class Scene: @unchecked Sendable {

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
                self.unmanagedTransformedPrimitives = transformedPrimitives.map { Unmanaged.passUnretained($0) }
                self.arena = arena
                self.meshesC = meshes.meshes.map { $0.cStruct }
        }

        func intersect(primId: PrimId, ray: Ray, tHit: inout Real) -> Bool {
                switch primId.type {
                case .triangle:
                        return Triangle(meshIndex: primId.id1, number: primId.id2).intersect(scene: self, ray: ray, tHit: &tHit) != nil
                case .geometricPrimitive:
                        return self.geometricPrimitives[primId.id1].intersect(scene: self, ray: ray, tHit: &tHit) != nil
                case .transformedPrimitive:
                        return self.unmanagedTransformedPrimitives[primId.id1]._withUnsafeGuaranteedRef {
                                $0.intersect(scene: self, ray: ray, tHit: &tHit) != nil
                        }
                case .areaLight:
                        return self.areaLights[primId.id1].intersect(scene: self, ray: ray, tHit: &tHit) != nil
                }
        }

        func getIntersectionData(
                primId: PrimId,
                ray: Ray,
                tHit: inout Real,
                data: inout TriangleIntersection
        ) -> Bool {
                switch primId.type {
                case .triangle:
                        return Triangle(meshIndex: primId.id1, number: primId.id2).getIntersectionData(scene: self, ray: ray, tHit: &tHit, data: &data)
                case .geometricPrimitive:
                        return self.geometricPrimitives[primId.id1].getIntersectionData(scene: self, ray: ray, tHit: &tHit, data: &data)
                case .transformedPrimitive:
                        return self.unmanagedTransformedPrimitives[primId.id1]._withUnsafeGuaranteedRef {
                                $0.getIntersectionData(scene: self, ray: ray, tHit: &tHit, data: &data)
                        }
                case .areaLight:
                        return self.areaLights[primId.id1].getIntersectionData(scene: self, ray: ray, tHit: &tHit, data: &data)
                }
        }

        func computeSurfaceInteraction(
                primId: PrimId,
                data: TriangleIntersection,
                worldRay: Ray
        ) -> SurfaceInteraction? {
                switch primId.type {
                case .triangle:
                        return Triangle(meshIndex: primId.id1, number: primId.id2).computeSurfaceInteraction(scene: self, data: data, worldRay: worldRay)
                case .geometricPrimitive:
                        return self.geometricPrimitives[primId.id1].computeSurfaceInteraction(scene: self, data: data, worldRay: worldRay)
                case .transformedPrimitive:
                        return self.unmanagedTransformedPrimitives[primId.id1]._withUnsafeGuaranteedRef {
                                $0.computeSurfaceInteraction(scene: self, data: data, worldRay: worldRay)
                        }
                case .areaLight:
                        return self.areaLights[primId.id1].computeSurfaceInteraction(scene: self, data: data, worldRay: worldRay)
                }
        }

        let lights: [Light]
        let infiniteLights: [InfiniteLight]
        let materials: [Material]
        let meshes: TriangleMeshes
        let geometricPrimitives: [GeometricPrimitive]
        let areaLights: [AreaLight]
        let transformedPrimitives: [TransformedPrimitive]
        let unmanagedTransformedPrimitives: [Unmanaged<TransformedPrimitive>]
        let arena: TextureArena
        let meshesC: [TriangleMesh_C]
}
