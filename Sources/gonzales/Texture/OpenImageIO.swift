import Foundation
import openimageio

func bla() {
        guard let textureSystem = OIIO.TextureSystem.create(true, nil) else {
                exit(0)
        }
        //let filename = OIIO.ustring("Skydome.pfm")
        //let filename = OIIO.ustring("bla.exr")
        //var options = OIIO.TextureOpt()
        //let s: Float = 0.5
        //let t: Float = 0.5
        //let dsdx: Float = 0.0
        //let dtdx: Float = 0.0
        //let dsdy: Float = 0.0
        //let dtdy: Float = 0.0
        //let nchannels: Int32 = 3
        //var result: [Float] = [0, 0, 0]
        //var successful = false
        //result.withUnsafeMutableBufferPointer { resultPointer in
        //        let resultAddress = resultPointer.baseAddress
        //        successful = textureSystem.pointee.texture(
        //                filename, &options, s, t, dsdx, dtdx, dsdy, dtdy,
        //                nchannels, resultAddress, nil, nil)
        //}
        //let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 3)

        // The following doesn't work because virtual methods are not supported by c++ interop
        // right now
        //successful = textureSystem.pointee.texture(
        //        filename, &options, s, t, dsdx, dtdx, dsdy, dtdy,
        //        nchannels, buffer, nil, nil)

        //if successful {
        //        print(result)
        //} else {
        //        print("Not successful.")
        //}
        OIIO.TextureSystem.destroy(textureSystem, true)
}
