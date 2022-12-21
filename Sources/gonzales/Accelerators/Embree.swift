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
                                case let triangle as Triangle:
                                        geometry(triangle: triangle)
                                        bounds = union(first: bounds, second: triangle.worldBound())
                                        materials[geomID] = geometricPrimitive.material
                                default:
                                        embreeError("Unknown shape in GeometricPrimitive.")
                                }
                        case let areaLight as AreaLight:
                                switch areaLight.shape {
                                case let triangle as Triangle:
                                        geometry(triangle: triangle)
                                        bounds = union(first: bounds, second: triangle.worldBound())
                                        areaLights[geomID] = areaLight
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

        func geometry(triangle: Triangle) {
                let points = triangle.getLocalPoints()
                let a = points.0
                let b = points.1
                let c = points.2
                embreeGeometry(
                        ax: a.x, ay: a.y, az: a.z, bx: b.x, by: b.y, bz: b.z, cx: c.x, cy: c.y,
                        cz: c.z)
        }

        var counter = 0

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) {
                var nx: FloatX = 0
                var ny: FloatX = 0
                var nz: FloatX = 0
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
                        nx = rayhit.hit.Ng_x
                        ny = rayhit.hit.Ng_y
                        nz = rayhit.hit.Ng_z
                        intersected = true
                }
                guard intersected else {
                        return
                }
                tHit = tout
                interaction.valid = true
                interaction.position = ray.origin + tout * ray.direction
                interaction.normal = normalized(Normal(x: nx, y: ny, z: nz))
                interaction.shadingNormal = interaction.normal
                interaction.wo = -ray.direction
                interaction.dpdu = up  // TODO
                interaction.faceIndex = 0  // TODO
                if let material = materials[geomID] {
                        interaction.material = material
                }
                if let areaLight = areaLights[geomID] {
                        interaction.areaLight = areaLight
                }
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        func embreeGeometry(
                ax: FloatX, ay: FloatX, az: FloatX,
                bx: FloatX, by: FloatX, bz: FloatX,
                cx: FloatX, cy: FloatX, cz: FloatX
        ) {
                guard let geom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE) else {
                        embreeError()
                }
                let floatSize = MemoryLayout<Float>.size
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
        var bounds = Bounds3f()
}
