import Foundation
import Security

/// Minimal Keychain seam so `AuthTokenStore` can be exercised against an
/// in-memory fake in unit tests -- the CI sandbox can block real Keychain
/// access, and we don't want that to gate `swift test`.
public protocol KeychainProtocol: Sendable {
    func setItem(account: String, service: String, data: Data) throws
    func readItem(account: String, service: String) throws -> Data?
    func deleteItem(account: String, service: String) throws
}

public enum KeychainError: Error, Equatable, Sendable {
    case unhandledStatus(OSStatus)
}

/// Real `kSecClassGenericPassword` backing, used by the app at runtime.
public final class SystemKeychain: KeychainProtocol, @unchecked Sendable {
    public static let shared = SystemKeychain()

    public init() {}

    public func setItem(account: String, service: String, data: Data) throws {
        let query = Self.query(account: account, service: service)
        // Overwrite semantics: delete-then-add avoids a separate update-vs-add
        // branch and keeps this a single well-understood code path.
        SecItemDelete(query as CFDictionary)
        var attributes = query
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        let status = SecItemAdd(attributes as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.unhandledStatus(status)
        }
    }

    public func readItem(account: String, service: String) throws -> Data? {
        var query = Self.query(account: account, service: service)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound {
            return nil
        }
        guard status == errSecSuccess else {
            throw KeychainError.unhandledStatus(status)
        }
        return result as? Data
    }

    public func deleteItem(account: String, service: String) throws {
        let status = SecItemDelete(Self.query(account: account, service: service) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unhandledStatus(status)
        }
    }

    private static func query(account: String, service: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecAttrService as String: service,
        ]
    }
}

/// In-memory fake for tests (and the CI-sandbox fallback): behaviorally
/// equivalent to `SystemKeychain` for `AuthTokenStore`'s purposes, with no
/// entitlement or device Keychain dependency.
public final class InMemoryKeychain: KeychainProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: Data] = [:]

    public init() {}

    public func setItem(account: String, service: String, data: Data) throws {
        lock.lock()
        defer { lock.unlock() }
        storage[Self.key(account: account, service: service)] = data
    }

    public func readItem(account: String, service: String) throws -> Data? {
        lock.lock()
        defer { lock.unlock() }
        return storage[Self.key(account: account, service: service)]
    }

    public func deleteItem(account: String, service: String) throws {
        lock.lock()
        defer { lock.unlock() }
        storage.removeValue(forKey: Self.key(account: account, service: service))
    }

    private static func key(account: String, service: String) -> String {
        "\(service)::\(account)"
    }
}

/// Keychain-backed access/refresh token storage. Access and refresh tokens
/// live in separate Keychain items (separate `account` values under one
/// `service`) per the INFRA-4 design.
public struct AuthTokenStore: Sendable {
    public static let defaultService = "com.dinkvision.pickleball.auth"

    private let keychain: KeychainProtocol
    private let service: String

    public init(keychain: KeychainProtocol = SystemKeychain.shared, service: String = AuthTokenStore.defaultService) {
        self.keychain = keychain
        self.service = service
    }

    private enum Account: String {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
    }

    public func storeAccessToken(_ token: String) throws {
        try keychain.setItem(account: Account.accessToken.rawValue, service: service, data: Data(token.utf8))
    }

    public func readAccessToken() throws -> String? {
        guard let data = try keychain.readItem(account: Account.accessToken.rawValue, service: service) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    public func deleteAccessToken() throws {
        try keychain.deleteItem(account: Account.accessToken.rawValue, service: service)
    }

    public func storeRefreshToken(_ token: String) throws {
        try keychain.setItem(account: Account.refreshToken.rawValue, service: service, data: Data(token.utf8))
    }

    public func readRefreshToken() throws -> String? {
        guard let data = try keychain.readItem(account: Account.refreshToken.rawValue, service: service) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    public func deleteRefreshToken() throws {
        try keychain.deleteItem(account: Account.refreshToken.rawValue, service: service)
    }

    /// Clears both items. Safe to call even if nothing is stored.
    public func clearAll() throws {
        try deleteAccessToken()
        try deleteRefreshToken()
    }

    /// Synchronous, best-effort presence check for launch-time gating.
    /// Swallows Keychain errors (treats them as "no session") rather than
    /// throwing, since callers use this for a UI branch, not a decision that
    /// needs to distinguish "empty" from "unreadable".
    public var hasAccessToken: Bool {
        if let token = try? readAccessToken() {
            return !token.isEmpty
        }
        return false
    }
}
