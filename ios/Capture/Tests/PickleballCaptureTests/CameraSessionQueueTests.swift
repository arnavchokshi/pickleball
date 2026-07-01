import Foundation
import XCTest
@testable import PickleballCapture

final class CameraSessionQueueTests: XCTestCase {
    func testRunExecutesWorkOffMainThread() async throws {
        let queue = CameraSessionQueue(label: "test.camera.session.off-main")

        let ranOnMainThread = try await queue.run {
            Thread.isMainThread
        }

        XCTAssertFalse(ranOnMainThread)
    }

    func testRunSerializesQueuedWork() async throws {
        let queue = CameraSessionQueue(label: "test.camera.session.serial")
        let tracker = ThreadSafeOverlapTracker()

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    try await queue.run {
                        tracker.enter()
                        Thread.sleep(forTimeInterval: 0.005)
                        tracker.leave()
                    }
                }
            }

            try await group.waitForAll()
        }

        XCTAssertEqual(tracker.maxConcurrentExecutions, 1)
    }
}

private final class ThreadSafeOverlapTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var activeExecutions = 0
    private(set) var maxConcurrentExecutions = 0

    func enter() {
        lock.withLock {
            activeExecutions += 1
            maxConcurrentExecutions = max(maxConcurrentExecutions, activeExecutions)
        }
    }

    func leave() {
        lock.withLock {
            activeExecutions -= 1
        }
    }
}
