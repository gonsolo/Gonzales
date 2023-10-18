import Foundation
import cuda
import cudaBridge

class Optix {

        private func add(triangle: Triangle) throws {
                let points = triangle.getWorldPoints()
                bounds = union(first: bounds, second: triangle.worldBound())
                gonzoAdd(
                        points.0.x, points.0.y, points.0.z,
                        points.1.x, points.1.y, points.1.z,
                        points.2.x, points.2.y, points.2.z)
        }

        func add(primitives: [Boundable & Intersectable]) throws {
                for primitive in primitives {
                        switch primitive {
                        case let geometricPrimitive as GeometricPrimitive:
                                switch geometricPrimitive.shape {
                                case let triangle as Triangle:
                                                try add(triangle: triangle)
                                                materials[triangleCount] = geometricPrimitive.material
                                                triangleCount += 1
                                default:
                                        var message = "Unknown shape in geometric primitive: "
                                        message += "\(geometricPrimitive.shape)"
                                        warnOnce(message)
                                }
                        case let areaLight as AreaLight:
                                switch areaLight.shape {
                                case let triangle as Triangle:
                                                try add(triangle: triangle)
                                                areaLights[triangleCount] = areaLight
                                                triangleCount += 1
                                default:
                                        fatalError("Unknown shape in AreaLight.")
                                }
                        default:
                                fatalError("Unknown primitive \(primitive).")
                        }
                }
        }

        func optixSetup() {
                gonzoSetup()
        }

        func optixRender() {
                var px: Float32 = 0
                var py: Float32 = 0
                var pz: Float32 = 0
                var nx: Float32 = 0
                var ny: Float32 = 0
                var nz: Float32 = 0
                var tHit: Float = 1e20
                var intersected: Int32 = 0
                var primID: Int32 = -1
                gonzoRender(false, 0, 0, 0, 0, 0, 0, &tHit, &px, &py, &pz, &nx, &ny, &nz, &intersected, &primID)
        }

        func optixRender(ray: Ray, tHit: inout Float) -> (Point, Normal, Bool, Int) {
                var px: Float32 = 0
                var py: Float32 = 0
                var pz: Float32 = 0
                var nx: Float32 = 0
                var ny: Float32 = 0
                var nz: Float32 = 0
                var intersected: Int32 = 0
                var primID32: Int32 = -1
                gonzoRender(
                        true,
                        ray.origin.x, ray.origin.y, ray.origin.z,
                        ray.direction.x, ray.direction.y, ray.direction.z,
                        &tHit,
                        &px, &py, &pz,
                        &nx, &ny, &nz,
                        &intersected,
                        &primID32
                        )
                let intersectionPoint = Point(x: px, y: py, z: pz)
                let intersectionNormal = Normal(x: nx, y: ny, z: nz)
                let intersectionIntersected: Bool = intersected == 1 ? true : false
                let primID = Int(primID32)
                return (intersectionPoint, intersectionNormal, intersectionIntersected, primID)
        }

        func render(ray: Ray, tHit: inout Float) throws -> (Point, Normal, Bool, Int) {
                return optixRender(ray: ray, tHit: &tHit);
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {
                let (intersectionPoint, intersectionNormal, intersected, primID) = try render(ray: ray, tHit: &tHit)
                if intersected {
                        interaction.valid = true
                        interaction.position = intersectionPoint
                        interaction.normal = intersectionNormal
                        interaction.shadingNormal = intersectionNormal
                        interaction.wo = -ray.direction
                        // dpdu
                        // uv
                        // faceIndex
                        interaction.material = materials[primID] ?? -1
                        interaction.areaLight = areaLights[primID] ?? nil
                }

        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        var bounds = Bounds3f()
        var triangleCount = 0
        var materials = [Int: MaterialIndex]()
        var areaLights = [Int: AreaLight]()
}

