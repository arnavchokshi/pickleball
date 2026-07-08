import Foundation

/// Wire shape of `server/routes/auth.py`'s `_issue_session` payload: always
/// `access_token`/`token_type`/`expires_in`; `refresh_token` is present only
/// when the request opted into the native contract (`login(native: true)`,
/// or `refresh` echoing a token that arrived via `X-Refresh-Token`).
public struct AuthSession: Codable, Equatable, Sendable {
    public var accessToken: String
    public var tokenType: String
    public var expiresIn: Int
    public var refreshToken: String?

    public init(accessToken: String, tokenType: String, expiresIn: Int, refreshToken: String? = nil) {
        self.accessToken = accessToken
        self.tokenType = tokenType
        self.expiresIn = expiresIn
        self.refreshToken = refreshToken
    }
}

/// `POST /api/auth/register` response shape.
public struct AuthRegisteredUser: Codable, Equatable, Sendable {
    public var id: String
    public var email: String
}

public enum AuthApiClientError: Error, Equatable, Sendable {
    case invalidResponse
    case httpStatus(Int, String)
    /// `refresh()`/`logout()` were called with nothing in the Keychain to
    /// send as `X-Refresh-Token`.
    case missingRefreshToken
}

/// Client for `/api/auth/*` (INFRA-1). Mirrors `RenderGatewayClient`'s
/// URLSession style: injectable session for tests, `RenderGatewayClient
/// .apiURL(path:baseURL:)` for path joining.
///
/// NATIVE REFRESH CONTRACT (confirmed against `server/routes/auth.py` at
/// lane start, 2026-07-07): `login` is always sent with `native: true`, so
/// the response body carries `refresh_token` in addition to the (ignored,
/// no cookie jar on iOS) `Set-Cookie`. `refresh`/`logout` send the stored
/// refresh token via the `X-Refresh-Token` header; the server's presence
/// check on that header is what re-arms `native` server-side and echoes a
/// rotated `refresh_token` back in the body.
public final class AuthApiClient: @unchecked Sendable {
    public static let defaultBaseURL = RenderGatewayClient.defaultBaseURL

    private let baseURL: URL
    private let session: URLSession
    private let tokenStore: AuthTokenStore

    public init(
        baseURL: URL = AuthApiClient.defaultBaseURL,
        session: URLSession = .shared,
        tokenStore: AuthTokenStore = AuthTokenStore()
    ) {
        self.baseURL = baseURL
        self.session = session
        self.tokenStore = tokenStore
    }

    @discardableResult
    public func register(email: String, password: String, inviteCode: String) async throws -> AuthRegisteredUser {
        let (data, _) = try await send(
            path: "/api/auth/register",
            jsonBody: ["email": email, "password": password, "invite_code": inviteCode]
        )
        let decoder = JSONDecoder()
        return try decoder.decode(AuthRegisteredUser.self, from: data)
    }

    /// Always requests the native refresh-token-in-body contract (`native:
    /// true`) -- iOS has no httpOnly cookie jar to fall back on.
    @discardableResult
    public func login(email: String, password: String, deviceLabel: String? = nil) async throws -> AuthSession {
        var body: [String: Any] = ["email": email, "password": password, "native": true]
        if let deviceLabel {
            body["device_label"] = deviceLabel
        }
        let (data, _) = try await send(path: "/api/auth/login", jsonBody: body)
        let session = try Self.decodeSession(data)
        try persist(session)
        return session
    }

    /// Rotates the refresh token: reads the current one from the Keychain,
    /// sends it via `X-Refresh-Token`, and persists whatever comes back.
    @discardableResult
    public func refresh() async throws -> AuthSession {
        guard let refreshToken = try tokenStore.readRefreshToken() else {
            throw AuthApiClientError.missingRefreshToken
        }
        let (data, _) = try await send(
            path: "/api/auth/refresh",
            jsonBody: nil,
            extraHeaders: ["X-Refresh-Token": refreshToken]
        )
        let session = try Self.decodeSession(data)
        try persist(session)
        return session
    }

    /// Revokes the refresh chain server-side (best-effort -- a failed
    /// network call still clears the local Keychain) and clears local
    /// tokens.
    public func logout() async throws {
        var extraHeaders: [String: String] = [:]
        if let refreshToken = try tokenStore.readRefreshToken() {
            extraHeaders["X-Refresh-Token"] = refreshToken
        }
        defer { try? tokenStore.clearAll() }
        _ = try await send(path: "/api/auth/logout", jsonBody: nil, extraHeaders: extraHeaders)
    }

    /// Current access token, if any -- for wiring into
    /// `RenderGatewayClient`'s / `PresignedUploadClient`'s bearer-token
    /// provider.
    public func currentAccessToken() -> String? {
        try? tokenStore.readAccessToken()
    }

    private func persist(_ session: AuthSession) throws {
        try tokenStore.storeAccessToken(session.accessToken)
        if let refreshToken = session.refreshToken {
            try tokenStore.storeRefreshToken(refreshToken)
        }
    }

    private func send(
        path: String,
        jsonBody: [String: Any]?,
        extraHeaders: [String: String] = [:]
    ) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: RenderGatewayClient.apiURL(path: path, baseURL: baseURL))
        request.httpMethod = "POST"
        for (key, value) in extraHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }
        if let jsonBody {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: jsonBody)
        }
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthApiClientError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw AuthApiClientError.httpStatus(httpResponse.statusCode, Self.errorDetail(from: data))
        }
        return (data, httpResponse)
    }

    private static func errorDetail(from data: Data) -> String {
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let detail = object["detail"] as? String {
            return detail
        }
        return String(data: data, encoding: .utf8) ?? "request failed"
    }

    private static func decodeSession(_ data: Data) throws -> AuthSession {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(AuthSession.self, from: data)
    }
}
