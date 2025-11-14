//@preconcurrency import Foundation  // fflush, stdout

import Foundation

actor ProgressReporter {
        let total: Int
        private var completed: Int = 0
        private let startTime = Date()

        init(total: Int) {
                self.total = total
        }

        func tileFinished() {
                completed += 1
        }

        func getProgressMetrics() -> (
                completed: Int, total: Int, timeElapsed: TimeInterval, averageTimePerTile: TimeInterval
        ) {
                let timeElapsed = Date().timeIntervalSince(startTime)
                let avg = completed > 0 ? timeElapsed / Double(completed) : 0

                return (completed: completed, total: total, timeElapsed: timeElapsed, averageTimePerTile: avg)
        }
}
