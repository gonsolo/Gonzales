import Foundation

struct Locker {

        func lock() { semaphore.wait() }

        func unlock() { semaphore.signal() }

        func locked(closure: () -> Void) {
                lock()
                defer { unlock() }
                closure()
        }

        let semaphore = DispatchSemaphore(value: 1)
}
