import XCTest
@testable import PickleballUpload

final class AuthTokenStoreTests: XCTestCase {
    func testAccessTokenRoundTripsThroughFakeKeychain() throws {
        let store = AuthTokenStore(keychain: InMemoryKeychain())

        XCTAssertNil(try store.readAccessToken())
        XCTAssertFalse(store.hasAccessToken)

        try store.storeAccessToken("access-123")

        XCTAssertEqual(try store.readAccessToken(), "access-123")
        XCTAssertTrue(store.hasAccessToken)

        try store.deleteAccessToken()

        XCTAssertNil(try store.readAccessToken())
        XCTAssertFalse(store.hasAccessToken)
    }

    func testRefreshTokenRoundTripsThroughFakeKeychainIndependentlyOfAccessToken() throws {
        let store = AuthTokenStore(keychain: InMemoryKeychain())

        try store.storeAccessToken("access-abc")
        try store.storeRefreshToken("refresh-xyz")

        XCTAssertEqual(try store.readAccessToken(), "access-abc")
        XCTAssertEqual(try store.readRefreshToken(), "refresh-xyz")

        try store.deleteRefreshToken()

        XCTAssertEqual(try store.readAccessToken(), "access-abc", "deleting the refresh token must not touch the access token")
        XCTAssertNil(try store.readRefreshToken())
    }

    func testOverwritingAnExistingTokenReplacesItRatherThanFailing() throws {
        let store = AuthTokenStore(keychain: InMemoryKeychain())

        try store.storeAccessToken("first")
        try store.storeAccessToken("second")

        XCTAssertEqual(try store.readAccessToken(), "second")
    }

    func testClearAllRemovesBothTokensAndIsSafeToCallWhenAlreadyEmpty() throws {
        let store = AuthTokenStore(keychain: InMemoryKeychain())
        try store.storeAccessToken("access-123")
        try store.storeRefreshToken("refresh-456")

        try store.clearAll()

        XCTAssertNil(try store.readAccessToken())
        XCTAssertNil(try store.readRefreshToken())

        // Calling again on an already-empty store must not throw.
        XCTAssertNoThrow(try store.clearAll())
    }

    func testTwoStoresOverTheSameKeychainAndServiceShareState() throws {
        let keychain = InMemoryKeychain()
        let writer = AuthTokenStore(keychain: keychain, service: "shared-service")
        let reader = AuthTokenStore(keychain: keychain, service: "shared-service")

        try writer.storeAccessToken("shared-token")

        XCTAssertEqual(try reader.readAccessToken(), "shared-token")
    }

    func testDistinctServicesDoNotLeakTokensBetweenEachOther() throws {
        let keychain = InMemoryKeychain()
        let storeA = AuthTokenStore(keychain: keychain, service: "service-a")
        let storeB = AuthTokenStore(keychain: keychain, service: "service-b")

        try storeA.storeAccessToken("token-a")

        XCTAssertEqual(try storeA.readAccessToken(), "token-a")
        XCTAssertNil(try storeB.readAccessToken())
    }
}
