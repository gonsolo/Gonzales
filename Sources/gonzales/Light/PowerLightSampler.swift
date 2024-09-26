import Foundation

// Adapted from https://github.com/kodecocodes/swift-algorithm-club/blob/master/Binary%20Search/BinarySearch.swift
public func lowerBound<T: Comparable>(_ a: [T], key: T) -> (Int, T) {
        var lowerBound = 0
        var upperBound = a.count
        while lowerBound < upperBound {
                let midIndex = lowerBound + (upperBound - lowerBound) / 2
                if a[midIndex] == key {
                        return (midIndex, a[midIndex])
                } else if a[midIndex] < key {
                        lowerBound = midIndex + 1
                } else {
                        upperBound = midIndex
                }
        }
        return (lowerBound, a[lowerBound])
}

// No async reduce in Swift as for now
extension Sequence {
        func asyncReduce<Result>(
                _ initialResult: Result,
                _ nextPartialResult: ((Result, Element) async throws -> Result)
        ) async rethrows -> Result {
                var result = initialResult
                for element in self {
                        result = try await nextPartialResult(result, element)
                }
                return result
        }
}

final class PowerLightSampler {

        @MainActor
        init(sampler: Sampler, lights: [Light]) async {
                self.sampler = sampler
                self.lights = lights
                totalPower = await lights.asyncReduce(0, { total, light in await total + light.power() })
                for (i, light) in lights.enumerated() {
                        if i == 0 {
                                await cumulativePowers.append(light.power())
                        } else {
                                await cumulativePowers.append(cumulativePowers.last! + light.power())
                        }
                }
        }

        @MainActor
        func chooseLight() async -> (Light, FloatX) {
                assert(lights.count > 0)
                let u = sampler.get1D()
                let powerIndex = u * totalPower
                let (i, _) = lowerBound(cumulativePowers, key: powerIndex)
                let light = lights[i]
                let probabilityDensity = await light.power() / totalPower
                return (light, probabilityDensity)
        }

        let sampler: Sampler
        let lights: [Light]
        var cumulativePowers: [FloatX] = []
        let totalPower: FloatX
}
