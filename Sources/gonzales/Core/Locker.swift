import Foundation

struct Locker {

        func lock() { semaphore.wait() }

        func unlock() { semaphore.signal() }

        func locked(closure: () -> Void) {
                lock()
                defer { unlock() }
                closure()
        }

        func lockedThrowing(closure: () throws -> Void) throws {
                lock()
                defer { unlock() }
                try closure()
        }

        let semaphore = DispatchSemaphore(value: 1)
}
