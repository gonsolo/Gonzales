import Foundation

actor ProgressReporter {
        let title: String
        let total: Int?
        private var completed: Int = 0
        private let startTime = Date()

        init(title: String, total: Int? = nil) {
                self.title = title
                self.total = total
        }

        func tileFinished() {
                completed += 1
        }
        
        func increment(by n: Int = 1) {
                completed += n
        }

        func getProgressMetrics() -> (
                total: Int?, completed: Int, timeElapsed: TimeInterval, averageTimePerTile: TimeInterval
        ) {
                let timeElapsed = Date().timeIntervalSince(startTime)
                let avg = completed > 0 ? timeElapsed / Double(completed) : 0
                return (total: total, completed: completed, timeElapsed: timeElapsed, averageTimePerTile: avg)
        }
}

func runProgressReporter(reporter: ProgressReporter) async {
        let reportInterval: UInt64 = 500_000_000 // 0.5s

        func formatTime(_ time: TimeInterval) -> String {
                let seconds = Int(time)
                let minutes = seconds / 60
                let remainingSeconds = seconds % 60
                if minutes > 0 {
                        return String(format: "%dm %02ds", minutes, remainingSeconds)
                } else {
                        return String(format: "%.1fs", Double(time))
                }
        }

        let title = reporter.title
        let spinner = ["|", "/", "-", "\\"]
        var spinIndex = 0

        while !Task.isCancelled {
                do {
                        try await Task.sleep(nanoseconds: reportInterval)

                        let metrics = await reporter.getProgressMetrics()
                        let progressString: String

                        if let total = metrics.total {
                                let percentage = total > 0 ? (Double(metrics.completed) / Double(total)) * 100.0 : 100.0
                                let predictedTotalTime = metrics.averageTimePerTile * Double(total)
                                progressString = String(
                                        format: "%@: %d / %d (%.1f%%) | Elapsed: %@ | Total Est.: %@",
                                        title,
                                        metrics.completed,
                                        total,
                                        percentage,
                                        formatTime(metrics.timeElapsed),
                                        formatTime(max(0, predictedTotalTime))
                                )
                        } else {
                                progressString = String(
                                        format: "%@ %@ | Elapsed: %@",
                                        title,
                                        spinner[spinIndex % spinner.count],
                                        formatTime(metrics.timeElapsed)
                                )
                                spinIndex += 1
                        }

                        FileHandle.standardOutput.write(Data((progressString + "\r").utf8))

                        if let total = metrics.total, metrics.completed >= total {
                                break
                        }
                } catch {
                        break
                }
        }

        let finalMetrics = await reporter.getProgressMetrics()
        let finalString: String
        if let total = finalMetrics.total {
                finalString = String(
                        format: "%@: %d / %d (100.0%%) - Complete. Total Time: %@\n",
                        title, total, total, formatTime(finalMetrics.timeElapsed)
                )
        } else {
                finalString = String(
                        format: "%@ Complete. Total Time: %@\n",
                        title, formatTime(finalMetrics.timeElapsed)
                )
        }
        FileHandle.standardOutput.write(Data(finalString.utf8))
}
