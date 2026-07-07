import Foundation

public enum CameraResourceOwner: Equatable, Sendable {
    case arKitSetup
    case avCapture
}

public enum CameraResourceOwnershipError: Error, Equatable, Sendable {
    case cameraAlreadyOwned(CameraResourceOwner)
}

public final class CameraResourceOwnership: @unchecked Sendable {
    private let lock = NSLock()
    private var owner: CameraResourceOwner?

    public init() {}

    public var activeOwner: CameraResourceOwner? {
        lock.lock()
        let current = owner
        lock.unlock()
        return current
    }

    public func beginARKitSetup() throws -> ARKitSetupCameraOwnershipToken {
        try begin(.arKitSetup)
        return ARKitSetupCameraOwnershipToken(ownership: self, owner: .arKitSetup)
    }

    public func beginAVCapture() throws -> AVCaptureCameraOwnershipToken {
        try begin(.avCapture)
        return AVCaptureCameraOwnershipToken(ownership: self, owner: .avCapture)
    }

    private func begin(_ requestedOwner: CameraResourceOwner) throws {
        lock.lock()
        defer {
            lock.unlock()
        }
        if let owner {
            throw CameraResourceOwnershipError.cameraAlreadyOwned(owner)
        }
        owner = requestedOwner
    }

    fileprivate func release(_ releasedOwner: CameraResourceOwner) {
        lock.lock()
        if owner == releasedOwner {
            owner = nil
        }
        lock.unlock()
    }
}

public final class ARKitSetupCameraOwnershipToken: @unchecked Sendable {
    private weak var ownership: CameraResourceOwnership?
    private let owner: CameraResourceOwner
    private let lock = NSLock()
    private var released = false

    fileprivate init(ownership: CameraResourceOwnership, owner: CameraResourceOwner) {
        self.ownership = ownership
        self.owner = owner
    }

    deinit {
        release()
    }

    public func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        let ownership = ownership
        lock.unlock()
        ownership?.release(owner)
    }
}

public final class AVCaptureCameraOwnershipToken: @unchecked Sendable {
    private weak var ownership: CameraResourceOwnership?
    private let owner: CameraResourceOwner
    private let lock = NSLock()
    private var released = false

    fileprivate init(ownership: CameraResourceOwnership, owner: CameraResourceOwner) {
        self.ownership = ownership
        self.owner = owner
    }

    deinit {
        release()
    }

    public func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        let ownership = ownership
        lock.unlock()
        ownership?.release(owner)
    }
}
