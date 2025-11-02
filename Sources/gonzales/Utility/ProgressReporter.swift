@preconcurrency import Foundation  // fflush, stdout

struct ProgressReporter {

        init() { total = 0 }

        init(total: Int) { self.total = total }

        mutating func update() {
                if current % frequency == 0 {
                        let percentage = 100 * current / total
                        print("\(percentage)% done", terminator: "\r")
                        fflush(stdout)
                }
                current += 1
        }

        mutating func reset() {
                current = 0
        }

        var current = 0
        let total: Int
        let frequency: Int = 10000
}
