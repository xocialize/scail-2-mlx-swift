// Foundation-only safetensors header reader (8-byte LE length + JSON) — the
// S0 key contract never touches MLX/Metal.
import Foundation

public enum SafetensorsHeader {
    public struct TensorInfo: Decodable {
        public let dtype: String
        public let shape: [Int]
    }

    public static func read(_ url: URL) throws -> [String: TensorInfo] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lenData = try handle.read(upToCount: 8), lenData.count == 8 else {
            throw POSIXError(.EIO)
        }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard let json = try handle.read(upToCount: Int(len)) else {
            throw POSIXError(.EIO)
        }
        var raw = try JSONSerialization.jsonObject(with: json) as? [String: Any] ?? [:]
        raw.removeValue(forKey: "__metadata__")
        let data = try JSONSerialization.data(withJSONObject: raw)
        return try JSONDecoder().decode([String: TensorInfo].self, from: data)
    }

    public static func keys(_ url: URL) throws -> Set<String> {
        Set(try read(url).keys)
    }
}
