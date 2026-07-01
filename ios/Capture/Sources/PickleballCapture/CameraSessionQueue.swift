import Foundation

public final class CameraSessionQueue: @unchecked Sendable {
    private let queue: DispatchQueue

    public init(label: String = "com.arnavchokshi.pickleball.camera-session") {
        queue = DispatchQueue(label: label, qos: .userInitiated)
    }

    public func run<Success: Sendable>(
        _ operation: @escaping @Sendable () throws -> Success
    ) async throws -> Success {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    continuation.resume(returning: try operation())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
