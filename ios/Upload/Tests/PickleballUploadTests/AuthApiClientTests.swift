import XCTest
@testable import PickleballUpload

// MARK: - Shared URLProtocol stub

/// Minimal request/response stub shared by every networked test in this
/// target (`AuthApiClientTests` + `PresignedUploadClientTests` compile into
/// the same `PickleballUploadTests` module, so this single definition is
/// visible from both without needing a separate test-support file).
struct StubbedResponse: Sendable {
    var statusCode: Int
    var headers: [String: String]
    var body: Data

    init(statusCode: Int, headers: [String: String] = [:], body: Data = Data()) {
        self.statusCode = statusCode
        self.headers = headers
        self.body = body
    }

    init(statusCode: Int, headers: [String: String] = [:], json: [String: Any]) {
        self.statusCode = statusCode
        self.headers = headers
        self.body = (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
    }
}

struct CapturedStubRequest: Sendable {
    var request: URLRequest
    var bodyData: Data?

    var bodyJSON: [String: Any]? {
        guard let bodyData else { return nil }
        return try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any]
    }
}

/// Thread-safe box backing `StubURLProtocol`'s class-level (effectively
/// global) state, so the type can satisfy Swift 6 strict concurrency without
/// pretending the mutable state isn't shared.
final class StubURLProtocolBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _handler: (@Sendable (URLRequest) throws -> StubbedResponse)?
    private var _captured: [CapturedStubRequest] = []

    var handler: (@Sendable (URLRequest) throws -> StubbedResponse)? {
        get { lock.lock(); defer { lock.unlock() }; return _handler }
        set { lock.lock(); defer { lock.unlock() }; _handler = newValue }
    }

    func capture(_ entry: CapturedStubRequest) {
        lock.lock()
        defer { lock.unlock() }
        _captured.append(entry)
    }

    var capturedRequests: [CapturedStubRequest] {
        lock.lock()
        defer { lock.unlock() }
        return _captured
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        _handler = nil
        _captured = []
    }
}

final class StubURLProtocol: URLProtocol {
    static let box = StubURLProtocolBox()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let bodyData = Self.resolvedBody(for: request)
        StubURLProtocol.box.capture(CapturedStubRequest(request: request, bodyData: bodyData))

        guard let handler = StubURLProtocol.box.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        do {
            let stubbed = try handler(request)
            let response = HTTPURLResponse(
                url: request.url ?? URL(string: "https://example.invalid")!,
                statusCode: stubbed.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: stubbed.headers
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: stubbed.body)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    /// Upload tasks (`session.upload(for:from:)`) surface their body via
    /// `httpBodyStream` rather than `httpBody` once intercepted by a custom
    /// protocol; read whichever is present.
    private static func resolvedBody(for request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 32 * 1024
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: bufferSize)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

func makeStubbedSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [StubURLProtocol.self]
    return URLSession(configuration: configuration)
}

// MARK: - AuthApiClient tests

final class AuthApiClientTests: XCTestCase {
    override func setUp() {
        super.setUp()
        StubURLProtocol.box.reset()
    }

    override func tearDown() {
        StubURLProtocol.box.reset()
        super.tearDown()
    }

    func testLoginStoresAccessAndRefreshTokensAndSendsNativeTrue() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/auth/login")
            return StubbedResponse(statusCode: 200, json: [
                "access_token": "access-abc",
                "token_type": "bearer",
                "expires_in": 900,
                "refresh_token": "refresh-xyz",
            ])
        }
        let keychain = InMemoryKeychain()
        let tokenStore = AuthTokenStore(keychain: keychain)
        let client = AuthApiClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession(), tokenStore: tokenStore)

        let session = try await client.login(email: "player@example.com", password: "hunter2222")

        XCTAssertEqual(session.accessToken, "access-abc")
        XCTAssertEqual(session.refreshToken, "refresh-xyz")
        XCTAssertEqual(session.expiresIn, 900)
        XCTAssertEqual(try tokenStore.readAccessToken(), "access-abc")
        XCTAssertEqual(try tokenStore.readRefreshToken(), "refresh-xyz")

        let captured = StubURLProtocol.box.capturedRequests
        XCTAssertEqual(captured.count, 1)
        let body = try XCTUnwrap(captured.first?.bodyJSON)
        XCTAssertEqual(body["email"] as? String, "player@example.com")
        XCTAssertEqual(body["native"] as? Bool, true)
        XCTAssertEqual(captured.first?.request.value(forHTTPHeaderField: "Content-Type"), "application/json")
    }

    func test401LoginSurfacesReadableErrorAndStoresNothing() async throws {
        StubURLProtocol.box.handler = { _ in
            StubbedResponse(statusCode: 401, json: ["detail": "invalid email or password"])
        }
        let tokenStore = AuthTokenStore(keychain: InMemoryKeychain())
        let client = AuthApiClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession(), tokenStore: tokenStore)

        do {
            _ = try await client.login(email: "player@example.com", password: "wrong-password")
            XCTFail("expected login to throw on 401")
        } catch let AuthApiClientError.httpStatus(status, detail) {
            XCTAssertEqual(status, 401)
            XCTAssertEqual(detail, "invalid email or password")
        }

        XCTAssertNil(try tokenStore.readAccessToken())
        XCTAssertNil(try tokenStore.readRefreshToken())
    }

    func testRegisterSendsInviteCodeAndDecodesCreatedUser() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/auth/register")
            return StubbedResponse(statusCode: 201, json: ["id": "user_abc123", "email": "player@example.com"])
        }
        let client = AuthApiClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            tokenStore: AuthTokenStore(keychain: InMemoryKeychain())
        )

        let user = try await client.register(email: "player@example.com", password: "hunter2222", inviteCode: "dink-invite")

        XCTAssertEqual(user.id, "user_abc123")
        XCTAssertEqual(user.email, "player@example.com")
        let body = try XCTUnwrap(StubURLProtocol.box.capturedRequests.first?.bodyJSON)
        XCTAssertEqual(body["invite_code"] as? String, "dink-invite")
    }

    func test403RegisterSurfacesInvalidInviteDetail() async throws {
        StubURLProtocol.box.handler = { _ in
            StubbedResponse(statusCode: 403, json: ["detail": "invalid invite code"])
        }
        let client = AuthApiClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            tokenStore: AuthTokenStore(keychain: InMemoryKeychain())
        )

        do {
            _ = try await client.register(email: "player@example.com", password: "hunter2222", inviteCode: "bad-code")
            XCTFail("expected register to throw on 403")
        } catch let AuthApiClientError.httpStatus(status, detail) {
            XCTAssertEqual(status, 403)
            XCTAssertEqual(detail, "invalid invite code")
        }
    }

    func testRefreshSendsStoredRefreshTokenAsHeaderAndRotatesBothTokens() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/auth/refresh")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-Refresh-Token"), "old-refresh")
            return StubbedResponse(statusCode: 200, json: [
                "access_token": "new-access",
                "token_type": "bearer",
                "expires_in": 900,
                "refresh_token": "new-refresh",
            ])
        }
        let tokenStore = AuthTokenStore(keychain: InMemoryKeychain())
        try tokenStore.storeAccessToken("stale-access")
        try tokenStore.storeRefreshToken("old-refresh")
        let client = AuthApiClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession(), tokenStore: tokenStore)

        let session = try await client.refresh()

        XCTAssertEqual(session.accessToken, "new-access")
        XCTAssertEqual(try tokenStore.readAccessToken(), "new-access")
        XCTAssertEqual(try tokenStore.readRefreshToken(), "new-refresh")
    }

    func testRefreshWithNoStoredTokenThrowsMissingRefreshTokenWithoutMakingARequest() async throws {
        StubURLProtocol.box.handler = { _ in
            XCTFail("refresh() must not hit the network with no stored refresh token")
            return StubbedResponse(statusCode: 500)
        }
        let client = AuthApiClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            tokenStore: AuthTokenStore(keychain: InMemoryKeychain())
        )

        do {
            _ = try await client.refresh()
            XCTFail("expected refresh() to throw")
        } catch AuthApiClientError.missingRefreshToken {
            // expected
        }
    }

    func testLogoutSendsRefreshTokenHeaderAndClearsLocalTokensEvenOnServerFailure() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/auth/logout")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-Refresh-Token"), "refresh-to-revoke")
            return StubbedResponse(statusCode: 500, body: Data("boom".utf8))
        }
        let tokenStore = AuthTokenStore(keychain: InMemoryKeychain())
        try tokenStore.storeAccessToken("access-abc")
        try tokenStore.storeRefreshToken("refresh-to-revoke")
        let client = AuthApiClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession(), tokenStore: tokenStore)

        do {
            try await client.logout()
            XCTFail("expected logout() to propagate the server error")
        } catch AuthApiClientError.httpStatus {
            // expected -- server failed, but...
        }

        XCTAssertNil(try tokenStore.readAccessToken(), "local tokens must clear even when the server call fails")
        XCTAssertNil(try tokenStore.readRefreshToken())
    }

    // MARK: - RenderGatewayClient auth header (shares this file's stub infra)

    func testRenderGatewayClientSetsAuthorizationHeaderWhenTokenProviderIsSet() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer bearer-token-123")
            return StubbedResponse(statusCode: 200, json: [
                "id": "job_1",
                "status": "running",
                "links": ["status": "/api/jobs/job_1"],
            ])
        }
        let client = RenderGatewayClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            accessTokenProvider: { "bearer-token-123" }
        )

        _ = try await client.fetchJobStatus("/api/jobs/job_1")

        XCTAssertEqual(StubURLProtocol.box.capturedRequests.count, 1)
    }

    func testRenderGatewayClientOmitsAuthorizationHeaderWhenNoTokenProviderIsSet() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
            return StubbedResponse(statusCode: 200, json: [
                "id": "job_1",
                "status": "running",
                "links": ["status": "/api/jobs/job_1"],
            ])
        }
        let client = RenderGatewayClient(baseURL: URL(string: "https://api.example.test")!, session: makeStubbedSession())

        _ = try await client.fetchJobStatus("/api/jobs/job_1")

        XCTAssertEqual(StubURLProtocol.box.capturedRequests.count, 1)
    }

    func testRenderGatewayClientOmitsAuthorizationHeaderWhenProviderReturnsNil() async throws {
        StubURLProtocol.box.handler = { request in
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
            return StubbedResponse(statusCode: 200, json: [
                "id": "job_1",
                "status": "running",
                "links": ["status": "/api/jobs/job_1"],
            ])
        }
        let client = RenderGatewayClient(
            baseURL: URL(string: "https://api.example.test")!,
            session: makeStubbedSession(),
            accessTokenProvider: { nil }
        )

        _ = try await client.fetchJobStatus("/api/jobs/job_1")

        XCTAssertEqual(StubURLProtocol.box.capturedRequests.count, 1)
    }
}
