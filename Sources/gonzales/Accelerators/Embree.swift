import Foundation
import embree3

final class Embree: Accelerator {

        init(primitives: inout [Boundable & Intersectable]) {
                rtcDevice = rtcNewDevice(nil)
                check(rtcDevice)
                rtcScene = rtcNewScene(rtcDevice)
                check(rtcScene)
                addPrimitives(primitives: &primitives)
                rtcCommitScene(rtcScene)
        }

        func check(_ pointer: Any?) {
                guard pointer != nil else {
                        embreeError()
                }
        }

        func addPrimitives(primitives: inout [Boundable & Intersectable]) {
                var geomID: UInt32 = 0
                for primitive in primitives {
                        switch primitive {
                        case let geometricPrimitive as GeometricPrimitive:
                                switch geometricPrimitive.shape {
                                case let curve as EmbreeCurve:
                                        geometry(curve: curve, geomID: geomID)
                                        bounds = union(first: bounds, second: curve.worldBound())
                                        materials[geomID] = geometricPrimitive.material
                                case let triangle as Triangle:
                                        geometry(triangle: triangle, geomID: geomID)
                                        bounds = union(first: bounds, second: triangle.worldBound())
                                        materials[geomID] = geometricPrimitive.material
                                default:
                                        embreeError("Unknown shape in geometric primitive.")
                                }
                        case let areaLight as AreaLight:
                                switch areaLight.shape {
                                case let triangle as Triangle:
                                        geometry(triangle: triangle, geomID: geomID)
                                        bounds = union(first: bounds, second: triangle.worldBound())
                                        areaLights[geomID] = areaLight
                                case let disk as Disk:
                                        _ = disk
                                        warnOnce("Ignoring disk in area light!")
                                case let sphere as Sphere:
                                        _ = sphere
                                        warnOnce("Ignoring sphere in area light!")
                                default:
                                        embreeError("Unknown shape in AreaLight.")
                                }
                        default:
                                embreeError("Unknown primitive.")
                        }
                        geomID += 1
                }
        }

        deinit {
                rtcReleaseScene(rtcScene)
                rtcReleaseDevice(rtcDevice)
        }

        func geometry(curve: EmbreeCurve, geomID: UInt32) {
                let points = curve.controlPoints
                embreeCurve(points: points, widths: curve.widths)
        }

        func geometry(triangle: Triangle, geomID: UInt32) {
                let points = triangle.getLocalPoints()
                let a = points.0
                let b = points.1
                let c = points.2
                embreeTriangle(
                        ax: a.x, ay: a.y, az: a.z, bx: b.x, by: b.y, bz: b.z, cx: c.x, cy: c.y,
                        cz: c.z)
                triangleMeshIndices[geomID] = triangle.meshIndex
                triangleIndices[geomID] = triangle.idx
        }

        var counter = 0

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) throws {

                let empty = { (line: Int) in
                        //print("No triangle intersection at line ", line)
                        return
                }

                var tout: FloatX = 0
                var geomID: UInt32 = 0

                let rtcInvalidGeometryId = UInt32.max

                var rayhit = RTCRayHit()
                rayhit.ray.org_x = ray.origin.x
                rayhit.ray.org_y = ray.origin.y
                rayhit.ray.org_z = ray.origin.z
                rayhit.ray.dir_x = ray.direction.x
                rayhit.ray.dir_y = ray.direction.y
                rayhit.ray.dir_z = ray.direction.z
                rayhit.ray.tnear = 0
                rayhit.ray.tfar = tHit
                rayhit.hit.geomID = rtcInvalidGeometryId

                var context = RTCIntersectContext()
                rtcInitIntersectContext(&context)

                rtcIntersect1(rtcScene, &context, &rayhit)

                var intersected = false
                if rayhit.hit.geomID != rtcInvalidGeometryId {
                        tout = rayhit.ray.tfar
                        geomID = rayhit.hit.geomID
                        intersected = true
                }
                guard intersected else {
                        return empty(#line)
                }
                interaction.valid = true
                interaction.position = ray.origin + tout * ray.direction
                interaction.normal = normalized(
                        Normal(
                                x: rayhit.hit.Ng_x,
                                y: rayhit.hit.Ng_y,
                                z: rayhit.hit.Ng_z))
                interaction.shadingNormal = interaction.normal
                interaction.wo = -ray.direction

                if triangleMeshIndices[geomID] != nil {
                        let triangle = try Triangle(
                                meshIndex: triangleMeshIndices[geomID]!,
                                number: triangleIndices[geomID]! / 3)

                        var p0t: Point = triangle.point0 - ray.origin
                        var p1t: Point = triangle.point1 - ray.origin
                        var p2t: Point = triangle.point2 - ray.origin

                        let kz = maxDimension(abs(ray.direction))
                        let kx = (kz + 1) % 3
                        let ky = (kx + 1) % 3
                        let d: Vector = permute(vector: ray.direction, x: kx, y: ky, z: kz)
                        p0t = permute(point: p0t, x: kx, y: ky, z: kz)
                        p1t = permute(point: p1t, x: kx, y: ky, z: kz)
                        p2t = permute(point: p2t, x: kx, y: ky, z: kz)

                        let sx: FloatX = -d.x / d.z
                        let sy: FloatX = -d.y / d.z
                        let sz: FloatX = 1.0 / d.z
                        p0t.x += sx * p0t.z
                        p0t.y += sy * p0t.z
                        p1t.x += sx * p1t.z
                        p1t.y += sy * p1t.z
                        p2t.x += sx * p2t.z
                        p2t.y += sy * p2t.z

                        let e0: FloatX = p1t.x * p2t.y - p1t.y * p2t.x
                        let e1: FloatX = p2t.x * p0t.y - p2t.y * p0t.x
                        let e2: FloatX = p0t.x * p1t.y - p0t.y * p1t.x

                        if (e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0) {
                                return empty(#line)
                        }
                        let det: FloatX = e0 + e1 + e2
                        if det == 0 {
                                return empty(#line)
                        }

                        p0t.z *= sz
                        p1t.z *= sz
                        p2t.z *= sz
                        let tScaled: FloatX = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z
                        if det < 0 && (tScaled >= 0 || tScaled < tHit * det) {
                                //print(det, tScaled, tHit, tHit * det)
                                return empty(#line)
                        } else if det > 0 && (tScaled <= 0 || tScaled > tHit * det) {
                                return empty(#line)
                        }

                        let invDet: FloatX = 1 / det
                        let b0: FloatX = e0 * invDet
                        let b1: FloatX = e1 * invDet
                        let b2: FloatX = e2 * invDet

                        let uv = triangleMeshes.getUVFor(
                                meshIndex: triangle.meshIndex,
                                indices: (
                                        triangle.vertexIndex0,
                                        triangle.vertexIndex1,
                                        triangle.vertexIndex2
                                )
                        )
                        let uvHit = triangle.computeUVHit(b0: b0, b1: b1, b2: b2, uv: uv)
                        interaction.uv = uvHit
                }
                // TODO: uv for curves
                let (dpdu, _) = makeCoordinateSystem(from: Vector(normal: interaction.normal))
                interaction.dpdu = dpdu

                interaction.faceIndex = 0  // TODO
                if let areaLight = areaLights[geomID] {
                        interaction.areaLight = areaLight
                }
                if let material = materials[geomID] {
                        interaction.material = material
                }

                tHit = tout
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        func embreeCurve(points: [Point], widths: (Float, Float)) {

                // TODO: Just width.0 used

                guard let geom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE)
                else {
                        embreeError()
                }
                let numCurves = points.count - 4
                var indices: [UInt32] = []
                for i in 0..<numCurves {
                        indices.append(UInt32(i))
                }
                rtcSetSharedGeometryBuffer(
                        geom,
                        RTC_BUFFER_TYPE_INDEX,
                        0,
                        RTC_FORMAT_UINT,
                        indices,
                        0,
                        unsignedIntSize,
                        numCurves)

                let numVertices = points.count
                guard
                        let vertices = rtcSetNewGeometryBuffer(
                                geom,
                                RTC_BUFFER_TYPE_VERTEX,
                                0,
                                RTC_FORMAT_FLOAT4,
                                vec4fSize,
                                numVertices)
                else {
                        embreeError()
                }
                for (counter, point) in points.enumerated() {
                        let xIndex = 4 * counter * floatSize
                        let yIndex = xIndex + floatSize
                        let zIndex = yIndex + floatSize
                        let wIndex = zIndex + floatSize
                        vertices.storeBytes(of: point.x, toByteOffset: xIndex, as: Float.self)
                        vertices.storeBytes(of: point.y, toByteOffset: yIndex, as: Float.self)
                        vertices.storeBytes(of: point.z, toByteOffset: zIndex, as: Float.self)
                        vertices.storeBytes(of: widths.0, toByteOffset: wIndex, as: Float.self)
                }
                rtcCommitGeometry(geom)
                rtcAttachGeometry(rtcScene, geom)
                rtcReleaseGeometry(geom)
        }

        func embreeTriangle(
                ax: FloatX, ay: FloatX, az: FloatX,
                bx: FloatX, by: FloatX, bz: FloatX,
                cx: FloatX, cy: FloatX, cz: FloatX
        ) {
                guard let geom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE) else {
                        embreeError()
                }
                guard
                        let vb = rtcSetNewGeometryBuffer(
                                geom,
                                RTC_BUFFER_TYPE_VERTEX,
                                0,
                                RTC_FORMAT_FLOAT3,
                                3 * floatSize,
                                3)
                else {
                        embreeError()
                }

                vb.storeBytes(of: ax, toByteOffset: 0 * floatSize, as: Float.self)
                vb.storeBytes(of: ay, toByteOffset: 1 * floatSize, as: Float.self)
                vb.storeBytes(of: az, toByteOffset: 2 * floatSize, as: Float.self)
                vb.storeBytes(of: bx, toByteOffset: 3 * floatSize, as: Float.self)
                vb.storeBytes(of: by, toByteOffset: 4 * floatSize, as: Float.self)
                vb.storeBytes(of: bz, toByteOffset: 5 * floatSize, as: Float.self)
                vb.storeBytes(of: cx, toByteOffset: 6 * floatSize, as: Float.self)
                vb.storeBytes(of: cy, toByteOffset: 7 * floatSize, as: Float.self)
                vb.storeBytes(of: cz, toByteOffset: 8 * floatSize, as: Float.self)

                let unsignedSize = MemoryLayout<UInt32>.size

                guard
                        let ib = rtcSetNewGeometryBuffer(
                                geom,
                                RTC_BUFFER_TYPE_INDEX,
                                0,
                                RTC_FORMAT_UINT3,
                                3 * unsignedSize,
                                1
                        )
                else {
                        embreeError()
                }
                ib.storeBytes(of: 0, toByteOffset: 0 * unsignedSize, as: UInt32.self)
                ib.storeBytes(of: 1, toByteOffset: 1 * unsignedSize, as: UInt32.self)
                ib.storeBytes(of: 2, toByteOffset: 2 * unsignedSize, as: UInt32.self)

                rtcCommitGeometry(geom)
                rtcAttachGeometry(rtcScene, geom)
                rtcReleaseGeometry(geom)
        }

        func embreeError(_ message: String = "") -> Never {
                print("embreeError: \(message)")
                exit(-1)
        }

        var rtcDevice: OpaquePointer?
        var rtcScene: OpaquePointer?
        var materials = [UInt32: MaterialIndex]()
        var areaLights = [UInt32: AreaLight]()
        var triangleMeshIndices = [UInt32: Int]()
        var triangleIndices = [UInt32: Int]()
        var bounds = Bounds3f()

        let floatSize = MemoryLayout<Float>.size
        let unsignedIntSize = MemoryLayout<UInt32>.size
        let vec4fSize = 4 * MemoryLayout<Float>.size
        let vec3faSize = 16 * MemoryLayout<Float>.size
}
