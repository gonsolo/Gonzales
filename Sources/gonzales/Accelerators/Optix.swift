import Foundation
import cuda
import cudaBridge

extension vec3f {
        init(point: Point) {
                self.init(point.x, point.y, point.z)
        }

        init(vector: Vector) {
                self.init(vector.x, vector.y, vector.z)
        }
}

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
                rays: [Ray],
                tHits: inout [FloatX],
                interactions: inout [SurfaceInteraction],
                skips: [Bool]
        ) throws {
                var ps = VectorVec3f()
                ps.resize(rays.count)
                var ns = VectorVec3f()
                ns.resize(rays.count)
                var intersecteds = VectorInt32()
                intersecteds.resize(rays.count)
                var primID32s = VectorInt32()
                primID32s.resize(rays.count)
                var tMaxs = VectorFloat()
                tMaxs.resize(rays.count)
                var skipsVector = VectorBool()
                for skip in skips {
                        skipsVector.push_back(skip)
                }
                let rayOrigins = rays.map { vec3f(point: $0.origin) }
                var rayOriginsVector = VectorVec3f()
                for origin in rayOrigins {
                        rayOriginsVector.push_back(origin)
                }
                let rayDirections = rays.map { vec3f(vector: $0.direction) }
                var rayDirectionsVector = VectorVec3f()
                for direction in rayDirections {
                        rayDirectionsVector.push_back(direction)
                }
                var tHitsVector = VectorFloat()
                for tHit in tHits {
                        tHitsVector.push_back(tHit)
                }
                var intersectionPoints = Array(repeating: Point(), count: rays.count)
                var intersectionNormals = Array(repeating: Normal(), count: rays.count)
                var intersectionIntersecteds = Array(repeating: false, count: rays.count)
                var primIDs = Array(repeating: 0, count: rays.count)

                optixIntersectVec(
                        rayOriginsVector,
                        rayDirectionsVector,
                        &tHitsVector,
                        &ps,
                        &ns,
                        &intersecteds,
                        &primID32s,
                        &tMaxs,
                        skipsVector)

                for i in 0..<tHitsVector.count {
                        tHits[i] = tHitsVector[i]
                }

                for i in 0..<rays.count {
                        if skips[i] {
                                continue
                        }
                        intersectionPoints[i] = Point(x: ps[i].x, y: ps[i].y, z: ps[i].z)
                        intersectionNormals[i] = Normal(x: ns[i].x, y: ns[i].y, z: ns[i].z)
                        intersectionIntersecteds[i] = intersecteds[i] == 1 ? true : false
                        primIDs[i] = Int(primID32s[i])

                        if intersectionIntersecteds[i] {
                                tHits[i] = tMaxs[i]
                                interactions[i].valid = true
                                interactions[i].position = intersectionPoints[i]
                                interactions[i].normal = intersectionNormals[i]
                                interactions[i].shadingNormal = intersectionNormals[i]
                                interactions[i].wo = -rays[i].direction
                                let (dpdu, _) = makeCoordinateSystem(
                                        from: Vector(normal: intersectionNormals[i]))
                                interactions[i].dpdu = dpdu
                                // TODO: uv
                                // TODO: faceIndex
                                interactions[i].material = materials[primIDs[i]] ?? -1
                                interactions[i].areaLight = areaLights[primIDs[i]] ?? nil
                        }
                }
        }

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction,
                skip: Bool = false
        ) throws {
                if skip {
                        return
                }
                var p = vec3f()
                var n = vec3f()
                var intersected: Int32 = 0
                var primID32: Int32 = -1
                var tMax: Float = 0
                let rayOrigin = vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
                let rayDirection = vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
                optixIntersect(
                        rayOrigin,
                        rayDirection,
                        &tHit,
                        &p,
                        &n,
                        &intersected,
                        &primID32,
                        &tMax,
                        false)
                let intersectionPoint = Point(x: p.x, y: p.y, z: p.z)
                let intersectionNormal = Normal(x: n.x, y: n.y, z: n.z)
                let intersectionIntersected: Bool = intersected == 1 ? true : false
                let primID = Int(primID32)
                if intersectionIntersected {
                        tHit = tMax
                        interaction.valid = true
                        interaction.position = intersectionPoint
                        interaction.normal = intersectionNormal
                        interaction.shadingNormal = intersectionNormal
                        interaction.wo = -ray.direction
                        let (dpdu, _) = makeCoordinateSystem(from: Vector(normal: intersectionNormal))
                        interaction.dpdu = dpdu
                        // TODO: uv
                        // TODO: faceIndex
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
