@preconcurrency import Foundation

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
