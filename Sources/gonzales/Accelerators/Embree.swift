import embree

import embree3

enum Embree3Error: Error {
        case device
}

var rtcDevice: OpaquePointer?
var rtcScene: OpaquePointer?

func embreeError() {
        print("embreeError")
}

func embreeInit() {
        rtcDevice = rtcNewDevice(nil)
        if rtcDevice == nil {
                embreeError()
        }
        //embreeSetDevice(rtcDevice)
        //embreeInitScene()
//        rtcScene = rtcNewScene(rtcDevice)
//        if rtcScene == nil {
//                embreeError()
//        }
}

//func embreeDeinit() {
//        rtcReleaseScene(rtcScene);
//        rtcReleaseDevice(rtcDevice);
//}
//
//func embreeCommit() {
//        rtcCommitScene(rtcScene);
//}
//
//func embreeGeometry(
//                ax: FloatX, ay: FloatX, az: FloatX,
//                bx: FloatX, by: FloatX, bz: FloatX,
//                cx: FloatX, cy: FloatX, cz: FloatX
//                ) {
//        let geom = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);
//        if geom == nil {
//                embreeError()
//        }
//        let floatSize = MemoryLayout<Float>.size
//        let vb = rtcSetNewGeometryBuffer(
//                        geom,
//                        RTC_BUFFER_TYPE_VERTEX,
//                        0,
//                        RTC_FORMAT_FLOAT3,
//                        3 * floatSize,
//                        3);
//        if vb == nil {
//                embreeError()
//        }
//
//        //let error = rtcGetDeviceError(rtcDevice)
//        //print("error \(error)")
//        //vb[0] = ax; vb[1] = ay; vb[2] = az; // 1st vertex
//        //vb[3] = bx; vb[4] = by; vb[5] = bz; // 2nd vertex
//        //vb[6] = cx; vb[7] = cy; vb[8] = cz; // 3rd vertex
//
//        vb?.storeBytes(of: ax, toByteOffset: 0 * floatSize, as: Float.self)
//        vb?.storeBytes(of: ay, toByteOffset: 1 * floatSize, as: Float.self)
//        vb?.storeBytes(of: az, toByteOffset: 2 * floatSize, as: Float.self)
//        vb?.storeBytes(of: bx, toByteOffset: 3 * floatSize, as: Float.self)
//        vb?.storeBytes(of: by, toByteOffset: 4 * floatSize, as: Float.self)
//        vb?.storeBytes(of: bz, toByteOffset: 5 * floatSize, as: Float.self)
//        vb?.storeBytes(of: cx, toByteOffset: 6 * floatSize, as: Float.self)
//        vb?.storeBytes(of: cy, toByteOffset: 7 * floatSize, as: Float.self)
//        vb?.storeBytes(of: cz, toByteOffset: 8 * floatSize, as: Float.self)
//
//        let unsignedSize = MemoryLayout<UInt32>.size
//
//        let ib = rtcSetNewGeometryBuffer(
//                        geom,
//                        RTC_BUFFER_TYPE_INDEX,
//                        0,
//                        RTC_FORMAT_UINT3,
//                        3 * unsignedSize,
//                        1);
//        if ib == nil {
//                embreeError()
//        }
//        //ib[0] = 0; ib[1] = 1; ib[2] = 2;
//        ib?.storeBytes(of: 0, toByteOffset: 0 * unsignedSize, as: UInt32.self)
//        ib?.storeBytes(of: 1, toByteOffset: 1 * unsignedSize, as: UInt32.self)
//        ib?.storeBytes(of: 2, toByteOffset: 2 * unsignedSize, as: UInt32.self)
//
//        rtcCommitGeometry(geom);
//        rtcAttachGeometry(rtcScene, geom);
//        rtcReleaseGeometry(geom);
//}
//
//func embreeIntersect(
//                ox: Float, oy: Float, oz: Float,
//                dx: Float, dy: Float, dz: Float,
//                tnear: Float, tfar: Float,
//                nx: inout Float, ny: inout Float, nz: inout Float,
//                tout: inout Float,
//                geomID: inout UInt32
//) -> Bool {
//        var rayhit =  RTCRayHit()
//        rayhit.ray.org_x = ox; rayhit.ray.org_y = oy; rayhit.ray.org_z = oz;
//        rayhit.ray.dir_x = dx; rayhit.ray.dir_y = dy; rayhit.ray.dir_z = dz;
//        rayhit.ray.tnear = tnear;
//        rayhit.ray.tfar = tfar;
//        let RTC_INVALID_GEOMETRY_ID = UInt32.max
//        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
//
//        var context = RTCIntersectContext()
//        rtcInitIntersectContext(&context);
//
//        rtcIntersect1(rtcScene, &context, &rayhit);
//
//        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
//                tout = rayhit.ray.tfar;
//                geomID = rayhit.hit.geomID;
//                nx = rayhit.hit.Ng_x;
//                ny = rayhit.hit.Ng_y;
//                nz = rayhit.hit.Ng_z;
//                return true;
//        } else {
//                return false;
//        }
//}

final class Embree: Accelerator {

         init(primitives: inout [Boundable & Intersectable]) {
                 embreeInit()
                 for primitive in primitives {
                         if let geometricPrimitive = primitive as? GeometricPrimitive {
                                 if let triangle = geometricPrimitive.shape as? Triangle {
                                         geometry(triangle: triangle)
                                         bounds = union(first: bounds, second: triangle.worldBound())
                                         materials.append(geometricPrimitive.material)
                                 }
                         }
                 }
                 //embreeCommit()
         }

         deinit {
                 //embreeDeinit()
         }

         func commit() {
                 //embreeCommit()
         }

         func geometry(triangle: Triangle) {
                //let points = triangle.getLocalPoints()
                //let a = points.0
                //let b = points.1
                //let c = points.2
                //embreeGeometry(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
                //embreeGeometry(ax: a.x, ay: a.y, az: a.z, bx: b.x, by: b.y, bz: b.z, cx: c.x, cy: c.y, cz: c.z)
        }

        var counter = 0

        func intersect(
                ray: Ray,
                tHit: inout FloatX,
                material: MaterialIndex,
                interaction: inout SurfaceInteraction
        ) {
                let nx: FloatX = 0
                let ny: FloatX = 0
                let nz: FloatX = 0
                let tout: FloatX = 0
                let geomID: UInt32 = 0
                //let intersected = embreeIntersect(
                //        ox: ray.origin.x, oy: ray.origin.y, oz: ray.origin.z,
                //        dx: ray.direction.x, dy: ray.direction.y, dz: ray.direction.z,
                //        tnear: 0.0, tfar: tHit, nx: &nx, ny: &ny, nz: &nz, tout: &tout, geomID: &geomID)
                //let intersected = embreeIntersect(
                //        ray.origin.x, ray.origin.y, ray.origin.z,
                //        ray.direction.x, ray.direction.y, ray.direction.z,
                //        0.0, tHit, &nx, &ny, &nz, &tout, &geomID)
                //guard intersected else {
                //        return
                //}
                tHit = tout
                interaction.valid = true
                interaction.position = ray.origin + tout * ray.direction
                interaction.normal = normalized(Normal(x: nx, y: ny, z: nz))
                interaction.shadingNormal = interaction.normal
                interaction.wo = -ray.direction
                interaction.dpdu = up  // TODO
                interaction.faceIndex = 0  // TODO
                interaction.material = materials[geomID]
        }

        func worldBound() -> Bounds3f {
                return bounds
        }

        func objectBound() -> Bounds3f {
                return bounds
        }

        var materials = [Int]()
        var bounds = Bounds3f()
}
