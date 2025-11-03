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
                _ nextPartialResult: (@Sendable (Result, Element) async throws -> Result)
        ) async rethrows -> Result {
                var result = initialResult
                for element in self {
                        result = try await nextPartialResult(result, element)
                }
                return result
        }
}

struct PowerLightSampler: Sendable {

        @MainActor
        init(sampler: RandomSampler, lights: [Light], scene: Scene) async {
                self.sampler = sampler
                self.lights = lights

                var cumulativePowers = [FloatX]()
                totalPower = await lights.asyncReduce(0, { total, light in total + light.power(scene: scene) })
                for (i, light) in lights.enumerated() {
                        if i == 0 {
                                cumulativePowers.append(light.power(scene: scene))
                        } else {
                                cumulativePowers.append(cumulativePowers.last! + light.power(scene: scene))
                        }
                }

                self.cumulativePowers = cumulativePowers
        }

        func chooseLight(scene: Scene) -> (Light, FloatX) {
                assert(lights.count > 0)
                let u = sampler.get1D()
                let powerIndex = u * totalPower
                let (i, _) = lowerBound(cumulativePowers, key: powerIndex)
                let light = lights[i]
                let probabilityDensity = light.power(scene: scene) / totalPower
                return (light, probabilityDensity)
        }

        let sampler: RandomSampler
        let lights: [Light]
        let cumulativePowers: [FloatX]
        let totalPower: FloatX
}
