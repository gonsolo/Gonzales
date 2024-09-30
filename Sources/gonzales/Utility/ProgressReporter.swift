import Foundation  // fflush, stdout

actor ProgressReporter {

        init() { total = 0 }

        init(total: Int) { self.total = total }

        func update() {
                if current % frequency == 0 {
                        let percentage = 100 * current / total
                        print("\(percentage)% done", terminator: "\r")
                        fflush(stdout)
                }
                current += 1
        }

        func reset() {
                current = 0
        }

        var current = 0
        let total: Int
        let frequency: Int = 10000
}
