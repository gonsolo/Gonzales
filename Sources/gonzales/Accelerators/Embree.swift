import Foundation
import embree4

// swift-format-ignore: AlwaysUseLowerCamelCase
@_silgen_name("_swift_stdlib_immortalize")
func _swift_stdlib_immortalize(_ p: UnsafeMutableRawPointer)

func immortalize(_ o: AnyObject) {
        withExtendedLifetime(o) {
                _swift_stdlib_immortalize(Unmanaged.passUnretained(o).toOpaque())
        }
}

let embree = Embree()

protocol EmbreeBase {

        func check(_ pointer: Any?)
        func embreeError(_ message: String) -> Never
}

extension EmbreeBase {

        func check(_ pointer: Any?) {
                guard pointer != nil else {
                        embreeError()
                }
        }

        func embreeError(_ message: String = "") -> Never {
                print("embreeError: \(message)")
                exit(-1)
        }

}

final class Embree: EmbreeBase {

        init() {
                rtcDevice = rtcNewDevice(nil)
                check(rtcDevice)
        }

        deinit {
                rtcReleaseDevice(rtcDevice)
        }

        var rtcDevice: OpaquePointer?
}

final class EmbreeAccelerator: EmbreeBase {

        init(
                bounds: Bounds3f,
                rtcScene: OpaquePointer?,
                materials: [UInt32: MaterialIndex],
                mediumInterfaces: [UInt32: MediumInterface?],
                areaLights: [UInt32: AreaLight],
                triangleUVs: [UInt32: (Vector2F, Vector2F, Vector2F)],
                instanceMap: [UInt32: AcceleratorIndex]
        ) {
                self.bounds = bounds
                self.rtcScene = rtcScene
                self.materials = materials
                self.mediumInterfaces = mediumInterfaces
                self.areaLights = areaLights
                self.triangleUVs = triangleUVs
                self.instanceMap = instanceMap
        }

        deinit {
                rtcReleaseScene(rtcScene)
        }

        @MainActor
        func worldBound() -> Bounds3f {
                return bounds
        }

        @MainActor
        func objectBound() -> Bounds3f {
                return bounds
        }

        @MainActor
        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) async throws {
                try intersect(
                        originX: ray.origin.x,
                        originY: ray.origin.y,
                        originZ: ray.origin.z,
                        directionX: ray.direction.x,
                        directionY: ray.direction.y,
                        directionZ: ray.direction.z,
                        tHit: &tHit,
                        interaction: &interaction)
        }

        @MainActor
        func intersect(
                originX: FloatX,
                originY: FloatX,
                originZ: FloatX,
                directionX: FloatX,
                directionY: FloatX,
                directionZ: FloatX,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {

                let empty = { (line: Int) in
                        //print("No triangle intersection at line ", line)
                        return
                }

                var tout: FloatX = 0
                var geomID: UInt32 = 0

                let rtcInvalidGeometryId = UInt32.max

                let rtcRay = RTCRay(
                        org_x: originX,
                        org_y: originY,
                        org_z: originZ,
                        tnear: 0,
                        dir_x: directionX,
                        dir_y: directionY,
                        dir_z: directionZ,
                        time: 0,
                        tfar: tHit,
                        mask: 0,
                        id: 0,
                        flags: 0)
                let rtcHit = RTCHit(
                        Ng_x: 0,
                        Ng_y: 0,
                        Ng_z: 0,
                        u: 0,
                        v: 0,
                        primID: 0,
                        geomID: rtcInvalidGeometryId,
                        instID: rtcInvalidGeometryId,
                        instPrimID: 0)

                var rayhit = RTCRayHit(ray: rtcRay, hit: rtcHit)
                rtcIntersect1(rtcScene, &rayhit, nil)

                guard rayhit.hit.geomID != rtcInvalidGeometryId else {
                        return empty(#line)
                }

                tout = rayhit.ray.tfar
                geomID = rayhit.hit.geomID

                var scene = self
                if rayhit.hit.instID != rtcInvalidGeometryId {
                        guard let acceleratorIndex = instanceMap[rayhit.hit.instID] else {
                                embreeError("No scene in instanceMap, instID: \(rayhit.hit.instID)")
                        }
                        let accelerator = accelerators[acceleratorIndex]
                        switch accelerator {
                        case .embree(let embree):
                                scene = embree
                        default:
                                fatalError("Embree expected!")
                        }
                }

                if let areaLight = scene.areaLights[geomID] {
                        if areaLight.alpha == 0 {
                                return empty(#line)
                        }
                }

                let bary1 = rayhit.hit.u
                let bary2 = rayhit.hit.v
                let bary0 = 1 - bary1 - bary2
                var uvs = (Vector2F(), Vector2F(), Vector2F())
                let uvsOpt = scene.triangleUVs[geomID]
                if uvsOpt == nil {
                        //let message = "TriangleUVs is nil: \(geomID)"
                        //embreeError(message)
                        //message += " in scene \(String(describing: rtcScene))"
                } else {
                        uvs = uvsOpt!
                }

                let uv = Point2F(
                        x: bary0 * uvs.0.x + bary1 * uvs.1.x + bary2 * uvs.2.x,
                        y: bary0 * uvs.0.y + bary1 * uvs.1.y + bary2 * uvs.2.y
                )
                interaction.valid = true
                interaction.position.x = originX + tout * directionX
                interaction.position.y = originY + tout * directionY
                interaction.position.z = originZ + tout * directionZ
                interaction.normal = normalized(
                        Normal(
                                x: rayhit.hit.Ng_x,
                                y: rayhit.hit.Ng_y,
                                z: rayhit.hit.Ng_z))
                interaction.shadingNormal = interaction.normal
                interaction.wo.x = -directionX
                interaction.wo.y = -directionY
                interaction.wo.z = -directionZ
                interaction.uv = uv

                let (dpdu, _) = makeCoordinateSystem(from: Vector(normal: interaction.normal))
                interaction.dpdu = dpdu

                interaction.faceIndex = 0  // TODO
                if let areaLight = scene.areaLights[geomID] {
                        interaction.areaLight = areaLight
                }
                if let material = scene.materials[geomID] {
                        interaction.material = material
                }
                if let mediumInterface = scene.mediumInterfaces[geomID] {
                        interaction.mediumInterface = mediumInterface
                }
                tHit = tout
        }

        @MainActor
        private var bounds = Bounds3f()

        var rtcScene: OpaquePointer?
        private var materials = [UInt32: MaterialIndex]()
        private var mediumInterfaces = [UInt32: MediumInterface?]()
        private var areaLights = [UInt32: AreaLight]()
        private var triangleUVs = [UInt32: (Vector2F, Vector2F, Vector2F)]()
        private var instanceMap = [UInt32: AcceleratorIndex]()
}

final class EmbreeBuilder: EmbreeBase {

        @MainActor
        init(primitives: [Boundable & Intersectable]) {
                rtcScene = rtcNewScene(embree.rtcDevice)
                check(rtcScene)
                addPrimitives(primitives: primitives)
        }

        @MainActor
        func getAccelerator() -> EmbreeAccelerator {
                let embreeAccelerator = EmbreeAccelerator(
                        bounds: bounds,
                        rtcScene: rtcScene,
                        materials: materials,
                        mediumInterfaces: mediumInterfaces,
                        areaLights: areaLights,
                        triangleUVs: triangleUVs,
                        instanceMap: instanceMap
                )
                immortalize(embreeAccelerator)
                return embreeAccelerator
        }

        private func setIDs(id: UInt32, primitive: GeometricPrimitive) {
                materials[id] = primitive.material
                if primitive.mediumInterface != nil {
                        mediumInterfaces[id] = primitive.mediumInterface
                }
        }

        @MainActor
        private func setBound(primitive: GeometricPrimitive) {
                bounds = union(first: bounds, second: primitive.worldBound())
        }

        @MainActor
        private func addPrimitives(primitives: [Boundable & Intersectable]) {
                for primitive in primitives {
                        switch primitive {
                        case let geometricPrimitive as GeometricPrimitive:
                                setIDs(id: geomID, primitive: geometricPrimitive)
                                setBound(primitive: geometricPrimitive)
                                switch geometricPrimitive.shape {
                                case let curve as EmbreeCurve:
                                        geometry(curve: curve, geomID: geomID)
                                case let triangle as Triangle:
                                        geometry(triangle: triangle, geomID: geomID)
                                case let sphere as Sphere:
                                        geometry(sphere: sphere, geomID: geomID)
                                case let disk as Disk:
                                        _ = disk
                                        warnOnce("Ignoring disk!")
                                default:
                                        var message = "Unknown shape in geometric primitive: "
                                        message += "\(geometricPrimitive.shape)"
                                        embreeError(message)
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
                        case let transformedPrimitive as TransformedPrimitive:
                                let acceleratorIndex = transformedPrimitive.acceleratorIndex
                                let accelerator = accelerators[acceleratorIndex]
                                var embreeAccelerator: EmbreeAccelerator
                                switch accelerator {
                                case .embree(let embree):
                                        embreeAccelerator = embree
                                case .boundingHierarchy:
                                        embreeError("Embree accelerator expected!")
                                //case .optix:
                                //        embreeError("Embree accelerator expected!")
                                }
                                let instance = rtcNewGeometry(embree.rtcDevice, RTC_GEOMETRY_TYPE_INSTANCE)
                                rtcSetGeometryInstancedScene(instance, embreeAccelerator.rtcScene)
                                rtcSetGeometryTimeStepCount(instance, 1)

                                rtcAttachGeometryByID(rtcScene, instance, geomID)

                                instanceMap[geomID] = acceleratorIndex

                                rtcReleaseGeometry(instance)

                                let transformMatrix = transformedPrimitive.transform.getMatrix()
                                let transposed = transformMatrix.transpose()
                                let xfm = transposed.backing.m2

                                let timeStep: UInt32 = 0
                                rtcSetGeometryTransform(
                                        instance,
                                        timeStep,
                                        RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
                                        xfm)

                                rtcCommitGeometry(instance)
                        //case let accelerator as Accelerator:
                        //        print("Accelerator")
                        //        switch accelerator {
                        //        case .embree:
                        //                print("Embree")
                        //        case .boundingHierarchy:
                        //                print("Bounding hierarchy")
                        //        }
                        default:
                                embreeError("Unknown primitive \(primitive).")
                        }
                        geomID += 1
                }
                rtcCommitScene(rtcScene)
        }

        func sizeInBytes<Key, Value>(of dictionary: [Key: Value]) -> Int {
                let keyStride = MemoryLayout<Key>.stride
                let valueStride = MemoryLayout<Value>.stride
                let bytes = (keyStride + valueStride) * 4 / 3 * dictionary.capacity
                return bytes
        }

        private func geometry(curve: EmbreeCurve, geomID: UInt32) {
                let points = curve.controlPoints
                embreeCurve(points: points, widths: curve.widths)
        }

        @MainActor
        private func geometry(triangle: Triangle, geomID: UInt32) {
                let points = triangle.getWorldPoints()
                let a = points.0
                let b = points.1
                let c = points.2

                let uv = triangleMeshes.getUVFor(
                        meshIndex: triangle.meshIndex,
                        indices: (triangle.vertexIndex0, triangle.vertexIndex1, triangle.vertexIndex2))
                embreeTriangle(
                        ax: a.x, ay: a.y, az: a.z, bx: b.x, by: b.y, bz: b.z, cx: c.x, cy: c.y,
                        cz: c.z)
                // TODO: Unused uvs shouldn't be set in the first place!
                if uv.0 != Vector2F() && uv.1 != Vector2F() && uv.2 != Vector2F() {
                        triangleUVs[geomID] = uv
                }
        }

        private func geometry(sphere: Sphere, geomID: UInt32) {
                let center = sphere.objectToWorld * Point()
                let radius = sphere.radius
                embreeSphere(center: center, radius: radius)
        }

        private func embreeCurve(points: [Point], widths: (Float, Float)) {

                // TODO: Just width.0 used

                guard
                        let geometry = rtcNewGeometry(
                                embree.rtcDevice,
                                RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE)
                else {
                        embreeError()
                }
                let numIndices = points.count - 3
                let indexSlot: UInt32 = 0
                guard
                        let indices = rtcSetNewGeometryBuffer(
                                geometry,
                                RTC_BUFFER_TYPE_INDEX,
                                indexSlot,
                                RTC_FORMAT_UINT,
                                unsignedIntSize,
                                numIndices)
                else {
                        embreeError()
                }
                for index in 0..<numIndices {
                        let offset = unsignedIntSize * index
                        indices.storeBytes(of: UInt32(index), toByteOffset: offset, as: UInt32.self)
                }

                let numVertices = points.count
                let vertexSlot: UInt32 = 0
                guard
                        let vertices = rtcSetNewGeometryBuffer(
                                geometry,
                                RTC_BUFFER_TYPE_VERTEX,
                                vertexSlot,
                                RTC_FORMAT_FLOAT4,
                                vec4fSize,
                                numVertices)
                else {
                        embreeError()
                }
                let curveRadius = widths.0 / 2
                for (counter, point) in points.enumerated() {
                        let xIndex = 4 * counter * floatSize
                        let yIndex = xIndex + floatSize
                        let zIndex = yIndex + floatSize
                        let wIndex = zIndex + floatSize
                        vertices.storeBytes(of: point.x, toByteOffset: xIndex, as: Float.self)
                        vertices.storeBytes(of: point.y, toByteOffset: yIndex, as: Float.self)
                        vertices.storeBytes(of: point.z, toByteOffset: zIndex, as: Float.self)
                        vertices.storeBytes(of: curveRadius, toByteOffset: wIndex, as: Float.self)
                }
                rtcCommitGeometry(geometry)
                rtcAttachGeometryByID(rtcScene, geometry, geomID)
                rtcReleaseGeometry(geometry)
        }

        private func embreeSphere(center: Point, radius: FloatX) {
                guard let geometry = rtcNewGeometry(embree.rtcDevice, RTC_GEOMETRY_TYPE_SPHERE_POINT) else {
                        embreeError()
                }
                let numberPoints = 1
                let slot: UInt32 = 0
                guard
                        let vertices = rtcSetNewGeometryBuffer(
                                geometry,
                                RTC_BUFFER_TYPE_VERTEX,
                                slot,
                                RTC_FORMAT_FLOAT4,
                                vec4fSize,
                                numberPoints)
                else {
                        embreeError()
                }
                vertices.storeBytes(of: center.x, toByteOffset: 0 * floatSize, as: Float.self)
                vertices.storeBytes(of: center.y, toByteOffset: 1 * floatSize, as: Float.self)
                vertices.storeBytes(of: center.z, toByteOffset: 2 * floatSize, as: Float.self)
                vertices.storeBytes(of: radius, toByteOffset: 3 * floatSize, as: Float.self)
                rtcCommitGeometry(geometry)
                rtcAttachGeometryByID(rtcScene, geometry, geomID)
                rtcReleaseGeometry(geometry)
        }

        private func embreeTriangle(
                ax: FloatX, ay: FloatX, az: FloatX,
                bx: FloatX, by: FloatX, bz: FloatX,
                cx: FloatX, cy: FloatX, cz: FloatX
        ) {
                guard let geometry = rtcNewGeometry(embree.rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE) else {
                        embreeError()
                }
                let indexSlot: UInt32 = 0
                let numberVertices = 3
                guard
                        let vertices = rtcSetNewGeometryBuffer(
                                geometry,
                                RTC_BUFFER_TYPE_VERTEX,
                                indexSlot,
                                RTC_FORMAT_FLOAT3,
                                vec3fSize,
                                numberVertices)
                else {
                        embreeError()
                }

                vertices.storeBytes(of: ax, toByteOffset: 0 * floatSize, as: Float.self)
                vertices.storeBytes(of: ay, toByteOffset: 1 * floatSize, as: Float.self)
                vertices.storeBytes(of: az, toByteOffset: 2 * floatSize, as: Float.self)
                vertices.storeBytes(of: bx, toByteOffset: 3 * floatSize, as: Float.self)
                vertices.storeBytes(of: by, toByteOffset: 4 * floatSize, as: Float.self)
                vertices.storeBytes(of: bz, toByteOffset: 5 * floatSize, as: Float.self)
                vertices.storeBytes(of: cx, toByteOffset: 6 * floatSize, as: Float.self)
                vertices.storeBytes(of: cy, toByteOffset: 7 * floatSize, as: Float.self)
                vertices.storeBytes(of: cz, toByteOffset: 8 * floatSize, as: Float.self)

                let unsignedSize = MemoryLayout<UInt32>.size
                let indicesSize = 3 * unsignedSize
                let slot: UInt32 = 0
                let numberIndices = 1
                guard
                        let indices = rtcSetNewGeometryBuffer(
                                geometry,
                                RTC_BUFFER_TYPE_INDEX,
                                slot,
                                RTC_FORMAT_UINT3,
                                indicesSize,
                                numberIndices
                        )
                else {
                        embreeError()
                }
                indices.storeBytes(of: 0, toByteOffset: 0 * unsignedSize, as: UInt32.self)
                indices.storeBytes(of: 1, toByteOffset: 1 * unsignedSize, as: UInt32.self)
                indices.storeBytes(of: 2, toByteOffset: 2 * unsignedSize, as: UInt32.self)

                rtcCommitGeometry(geometry)
                rtcAttachGeometryByID(rtcScene, geometry, geomID)
                rtcReleaseGeometry(geometry)
        }

        private var rtcScene: OpaquePointer?

        private var materials = [UInt32: MaterialIndex]()
        private var mediumInterfaces = [UInt32: MediumInterface?]()
        private var areaLights = [UInt32: AreaLight]()
        private var triangleUVs = [UInt32: (Vector2F, Vector2F, Vector2F)]()
        private var instanceMap = [UInt32: AcceleratorIndex]()

        @MainActor
        private var bounds = Bounds3f()

        private let floatSize = MemoryLayout<Float>.size
        private let unsignedIntSize = MemoryLayout<UInt32>.size
        private let vec4fSize = 4 * MemoryLayout<Float>.size
        private let vec3fSize = 3 * MemoryLayout<Float>.size
        private let vec3faSize = 16 * MemoryLayout<Float>.size

        private var geomID: UInt32 = 0
}
