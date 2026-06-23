// Image / video decode + mp4 encode for the --generate CLI. Mirrors the
// oracle scripts/generate.py input prep (resize_for_rectangle_crop, reshape
// mode 'center', PIL bicubic ≈ CGContext .high) and the family FrameEncode
// (H.264 mp4 via AVAssetWriter). Pure CoreGraphics/AVFoundation — parity is
// perceptual here (resize is not bit-equal to torchvision/PIL; the parity
// gates inject preprocessed tensors and bypass this path).
import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

enum MediaIOError: Error {
    case imageDecode(String)
    case videoNoFrames(String)
    case writerSetup(String)
    case writeIncomplete(String)
}

/// Render a CGImage into an exact w×h RGBA top-down raster (.high ≈ bicubic).
private func renderRGBA(_ cg: CGImage, _ w: Int, _ h: Int) -> [UInt8] {
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8,
        bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    // Drawing a CGImage straight into a premultipliedLast context yields the
    // top-down raster the rest of the pipeline uses (family FrameDecode W8).
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return rgba
}

/// resize_for_rectangle_crop (reshape_mode='center'): aspect-preserving resize
/// to cover, then center-crop to (targetW, targetH). Returns CHW float in
/// [-1,1] (length 3·targetH·targetW), row 0 = image top.
private func cropCHW(_ cg: CGImage, targetH: Int, targetW: Int) -> [Float] {
    let (sw, sh) = (cg.width, cg.height)
    let newH: Int, newW: Int
    if Double(sw) / Double(sh) > Double(targetW) / Double(targetH) {
        newH = targetH
        newW = Int((Double(sw) * Double(targetH) / Double(sh)).rounded())
    } else {
        newW = targetW
        newH = Int((Double(sh) * Double(targetW) / Double(sw)).rounded())
    }
    let rgba = renderRGBA(cg, newW, newH)  // top-down [newH,newW,4]
    let top = (newH - targetH) / 2
    let left = (newW - targetW) / 2

    let plane = targetH * targetW
    var chw = [Float](repeating: 0, count: 3 * plane)
    for y in 0..<targetH {
        for x in 0..<targetW {
            let sp = ((y + top) * newW + (x + left)) * 4
            let i = y * targetW + x
            chw[i] = Float(rgba[sp]) / 127.5 - 1            // R
            chw[plane + i] = Float(rgba[sp + 1]) / 127.5 - 1  // G
            chw[2 * plane + i] = Float(rgba[sp + 2]) / 127.5 - 1  // B
        }
    }
    return chw
}

/// Reference image → [3, H, W] in [-1, 1] (matches generate.py _load_image).
func loadImageCHW(_ path: String, targetH: Int, targetW: Int) throws -> MLXArray {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw MediaIOError.imageDecode(path) }
    let chw = cropCHW(cg, targetH: targetH, targetW: targetW)
    return MLXArray(chw, [3, targetH, targetW])
}

/// Video → [T, 3, H, W] in [-1, 1] (matches generate.py _load_video, T-C-H-W).
func loadVideoTCHW(
    _ path: String, targetH: Int, targetW: Int, maxFrames: Int? = nil
) async throws -> MLXArray {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let reader = try AVAssetReader(asset: asset)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        throw MediaIOError.videoNoFrames(path)
    }
    let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    reader.startReading()

    let ci = CIContext()
    var frames: [[Float]] = []
    while let sample = output.copyNextSampleBuffer(),
          let pb = CMSampleBufferGetImageBuffer(sample) {
        let img = CIImage(cvPixelBuffer: pb)
        guard let cg = ci.createCGImage(img, from: img.extent) else { continue }
        frames.append(cropCHW(cg, targetH: targetH, targetW: targetW))
        if let m = maxFrames, frames.count >= m { break }
    }
    guard !frames.isEmpty else { throw MediaIOError.videoNoFrames(path) }

    let t = frames.count
    let plane = targetH * targetW
    var tchw = [Float](repeating: 0, count: t * 3 * plane)
    for (i, f) in frames.enumerated() {
        let base = i * 3 * plane
        for j in 0..<(3 * plane) { tchw[base + j] = f[j] }
    }
    return MLXArray(tchw, [t, 3, targetH, targetW])
}

/// Frame tensor [3,H,W] in [-1,1] → interleaved RGB bytes (family rgbBytes).
private func rgbBytes(_ frame: MLXArray) -> ([UInt8], Int, Int) {
    let h = frame.dim(1), w = frame.dim(2)
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255).asType(.uint8).transposed(1, 2, 0)
    eval(rgb)
    return (rgb.asArray(UInt8.self), w, h)
}

/// Encode [3, T, H, W] in [-1, 1] as an H.264 mp4 at `fps` to `url`.
func writeMP4(frames: MLXArray, to url: URL, fps: Double) async throws {
    let t = frames.dim(1), h = frames.dim(2), w = frames.dim(3)
    if FileManager.default.fileExists(atPath: url.path) {
        try FileManager.default.removeItem(at: url)
    }
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
        ])
    guard writer.canAdd(input) else { throw MediaIOError.writerSetup("cannot add input") }
    writer.add(input)
    guard writer.startWriting() else {
        throw MediaIOError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let timescale = CMTimeScale(600)
    let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: timescale)
    for i in 0..<t {
        let (bytes, fw, fh) = rgbBytes(frames[0..., i])
        guard let pool = adaptor.pixelBufferPool else {
            throw MediaIOError.writerSetup("no pixel buffer pool")
        }
        var pbOut: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pbOut)
        guard let buffer = pbOut else { throw MediaIOError.writerSetup("pixel buffer alloc") }
        CVPixelBufferLockBaseAddress(buffer, [])
        let baseAddr = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<fh {
            for x in 0..<fw {
                let s = (y * fw + x) * 3, d = y * stride + x * 4
                baseAddr[d + 0] = bytes[s + 2]  // B
                baseAddr[d + 1] = bytes[s + 1]  // G
                baseAddr[d + 2] = bytes[s + 0]  // R
                baseAddr[d + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
        guard adaptor.append(buffer, withPresentationTime:
                CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
            throw MediaIOError.writeIncomplete("append failed at frame \(i), status=\(writer.status.rawValue)")
        }
    }
    input.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed, FileManager.default.fileExists(atPath: url.path) else {
        throw MediaIOError.writeIncomplete(
            "status=\(writer.status.rawValue) err=\(String(describing: writer.error))")
    }
}
