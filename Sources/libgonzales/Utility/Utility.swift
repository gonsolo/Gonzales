import Foundation  // log2, Process, URL, Pipe, Thread

let machineEpsilon = FloatX.ulpOfOne
let shadowEpsilon: FloatX = 0.00001
let oneMinusEpsilon: FloatX = 0.99999999999999989

public struct RenderOptions: Sendable {
        public var singleRayCoordinate = Point2i()
        public var quick = false
        public var singleRay = false
        public var justParse = false
        public var ptexMemory = 4  // GB
        public var sceneDirectory = String()
}

@MainActor
public var renderOptions = RenderOptions()

func radians(deg: FloatX) -> FloatX {
        return (FloatX.pi / 180) * deg
}

func gamma(count: Int) -> FloatX {
        return (FloatX(count) * machineEpsilon) / (1 - FloatX(count) * machineEpsilon)
}

func clamp<T: Comparable>(value: T, low: T, high: T) -> T {
        if value < low { return low } else if value > high { return high } else { return value }
}

func roundUpPower2(value: Int) -> Int {
        var value = value - 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        value |= value >> 32
        return value + 1
}

func log2Int(value: Int) -> UInt {
        return UInt(log2(FloatX(value)))
}

func gammaLinearToSrgb(value: FloatX) -> FloatX {
        if value <= 0.0031308 {
                return value / 12.92
        } else {
                return 1.055 * pow(value, 1.0 / 2.4) - 0.055
        }
}

func square(_ x: FloatX) -> FloatX { return x * x }

func square(_ x: RgbSpectrum) -> RgbSpectrum { return x * x }

func gammaSrgbToLinear(value: FloatX) -> FloatX {
        if value <= 0.04045 {
                return value / 12.92
        } else {
                return pow((value + 0.055) / 1.055, 2.4)
        }
}

func lerp(with t: FloatX, between first: FloatX, and second: FloatX) -> FloatX {
        return (1 - t) * first + t * second
}

func lerp(with t: RgbSpectrum, between first: RgbSpectrum, and second: RgbSpectrum) -> RgbSpectrum {
        return (white - t) * first + t * second
}

func lerp(with t: FloatX, between first: RgbSpectrum, and second: RgbSpectrum) -> RgbSpectrum {
        let t = RgbSpectrum(intensity: t)
        return (white - t) * first + t * second
}

@available(macOS 10.13, *)
func shell(_ launchPath: String, _ arguments: [String] = []) -> (String?, Int32) {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: launchPath)
        task.arguments = arguments
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
                try task.run()
        } catch {
                print("Error: \(error.localizedDescription)")
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)
        task.waitUntilExit()
        return (output, task.terminationStatus)
}

func demangle(symbol: String) -> String {
        if #available(macOS 10.13, *) {
                let demangle = "swift demangle"
                let (output, _) = shell(demangle, ["-compact", symbol])
                return (output!).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
                return symbol
        }
}

func printStack() {
        for symbol in Thread.callStackSymbols {
                guard let dollar = symbol.firstIndex(of: Character("$")) else { return }
                let start = symbol.index(after: dollar)
                guard let plus = symbol.firstIndex(of: Character("+")) else { return }
                let symbol = String(symbol[start..<plus])
                let demangled = demangle(symbol: symbol)
                print(demangled)
        }
}
