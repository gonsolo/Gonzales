@preconcurrency import Foundation

extension TimeInterval {
        public var humanReadable: String {
                let seconds = self.truncatingRemainder(dividingBy: 60)
                if self > 60 {
                        let minutes = self / 60
                        let minutesString = String(format: "%.0f", minutes)
                        let s = String(format: "%.0f", seconds)
                        return "\(minutesString)m\(s)s"
                } else {
                        let s = String(format: "%.1f", seconds)
                        return "\(s)s"
                }
        }
}

final class Timer {

        @MainActor
        init(_ name: String, newline: Bool = true) {
                self.name = name
                if newline {
                        print(name)
                } else {
                        print(name, terminator: "")
                }
                fflush(stdout)
                startTime = Date()
        }

        func stop() {
                endTime = Date()
        }

        var duration: TimeInterval {
                stop()
                guard #available(macOS 10.12, *) else { fatalError() }
                return DateInterval(start: startTime, end: endTime!).duration
        }

        var elapsed: String {
                return "\(duration.humanReadable)"
        }

        var name: String
        var startTime: Date
        var endTime: Date?
}
