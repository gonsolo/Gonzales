import Foundation

// Sobol Matrix Generator
// Generates SobolMatrices.swift from Joe-Kuo direction numbers.
// Reference: Joe & Kuo, "Constructing Sobol Sequences with Better
// Two-Dimensional Projections", SIAM J. Sci. Comput. 30(5), 2008.

let nDimensions = 1024
let matrixSize = 52  // bits of precision (covers double mantissa)

func generateSobolMatrices(directionNumbersPath: String) throws -> [[UInt32]] {
    let content = try String(contentsOfFile: directionNumbersPath, encoding: .utf8)
    let lines = content.components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty && !$0.starts(with: "d") }

    // Parse direction numbers: d s a m_i...
    struct DirectionEntry {
        let s: Int        // degree of primitive polynomial
        let a: Int        // polynomial coefficients
        let m: [Int]      // initial direction numbers
    }

    var entries: [DirectionEntry] = []
    for line in lines {
        let parts = line.split(separator: "\t").flatMap { $0.split(separator: " ") }
        guard parts.count >= 3 else { continue }
        // d, s, a, m_1, m_2, ..., m_s
        let s = Int(parts[1])!
        let a = Int(parts[2])!
        let mValues = parts[3...].map { Int($0)! }
        entries.append(DirectionEntry(s: s, a: a, m: mValues))
    }

    var matrices = [[UInt32]](repeating: [UInt32](repeating: 0, count: matrixSize), count: nDimensions)

    // Dimension 0: identity matrix (Van der Corput sequence)
    for i in 0..<32 {
        matrices[0][i] = 1 << (31 - i)
    }
    // Bits 32-51 are zero for dimension 0

    // Dimension 1: Van der Corput in base 2 with different scrambling
    // Generate from first entry
    for dim in 1..<nDimensions {
        guard dim - 1 < entries.count else { break }
        let entry = entries[dim - 1]
        let s = entry.s
        let a = entry.a

        // Compute direction numbers v[1..matrixSize]
        var v = [UInt32](repeating: 0, count: matrixSize + 1)

        // Initialize first s direction numbers from Joe-Kuo data
        for i in 1...s {
            v[i] = UInt32(entry.m[i - 1]) << (32 - i)
        }

        // Compute remaining direction numbers via recurrence
        for i in (s + 1)...matrixSize {
            v[i] = v[i - s] ^ (v[i - s] >> s)
            for j in 1..<s {
                if (a >> (s - 1 - j)) & 1 == 1 {
                    v[i] ^= v[i - j]
                }
            }
        }

        // Store as matrix columns
        for i in 0..<matrixSize {
            matrices[dim][i] = v[i + 1]
        }
    }

    return matrices
}

func formatSwiftSource(matrices: [[UInt32]]) -> String {
    var output = ""
    output += "import Foundation\n\n"
    output += "public let NSobolDimensions = \(nDimensions)\n"
    output += "public let SobolMatrixSize = \(matrixSize)\n"
    output += "\n"
    output += "public let sobolMatrices: [UInt32] = {\n"

    var data = Data()
    data.reserveCapacity(nDimensions * matrixSize * 4)
    for dim in 0..<nDimensions {
        for row in 0..<matrixSize {
            var value = matrices[dim][row].littleEndian
            withUnsafeBytes(of: &value) { buffer in
                data.append(contentsOf: buffer)
            }
        }
    }
    let base64 = data.base64EncodedString()

    output += "    let base64 = \"\(base64)\"\n"
    output += "    let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters)!\n"
    output += "    var arr = [UInt32](repeating: 0, count: \(nDimensions) * \(matrixSize))\n"
    output += "    arr.withUnsafeMutableBufferPointer { dest in\n"
    output += "        let _ = data.copyBytes(to: dest)\n"
    output += "    }\n"
    output += "    #if _endian(big)\n"
    output += "    for i in 0..<arr.count {\n"
    output += "        arr[i] = UInt32(littleEndian: arr[i])\n"
    output += "    }\n"
    output += "    #endif\n"
    output += "    return arr\n"
    output += "}()\n"

    return output
}

// Main
func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

guard CommandLine.arguments.count >= 3 else {
    printError("Usage: SobolGenerator <direction-numbers-file> <output-file>")
    exit(1)
}

let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]

do {
    let matrices = try generateSobolMatrices(directionNumbersPath: inputPath)
    let source = formatSwiftSource(matrices: matrices)
    try source.write(toFile: outputPath, atomically: true, encoding: .utf8)
    printError("Generated \(outputPath) (\(nDimensions) dimensions, \(matrixSize) matrix size)")
} catch {
    printError("Error: \(error)")
    exit(1)
}
