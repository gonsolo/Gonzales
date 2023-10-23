import Foundation
import cuda
import cudaBridge

class Optix {

        init(primitives: [Boundable & Intersectable]) throws {
                try add(primitives: primitives)
                optixSetup()
        }

        private func add(triangle: Triangle) throws {
                let points = triangle.getWorldPoints()
                bounds = union(first: bounds, second: triangle.worldBound())
                let a = vec3f(points.0.x, points.0.y, points.0.z)
                let b = vec3f(points.1.x, points.1.y, points.1.z)
                let c = vec3f(points.2.x, points.2.y, points.2.z)
                optixAddTriangle(a, b, c)
        }

        private func add(primitives: [Boundable & Intersectable]) throws {
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

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                var p = vec3f()
                var n = vec3f()
                var intersected: Int32 = 0
                var primID32: Int32 = -1
                let rayOrigin = vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
                let rayDirection = vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
                optixIntersect(
                        rayOrigin,
                        rayDirection,
                        &tHit,
                        &p,
                        &n,
                        &intersected,
                        &primID32)
                let intersectionPoint = Point(x: p.x, y: p.y, z: p.z)
                let intersectionNormal = Normal(x: n.x, y: n.y, z: n.z)
                let intersectionIntersected: Bool = intersected == 1 ? true : false
                let primID = Int(primID32)
                if intersectionIntersected {
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
