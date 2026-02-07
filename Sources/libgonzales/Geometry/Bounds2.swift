public struct Bounds2<T: Sendable>: Sendable {

        public init(pMin: Point2<T>, pMax: Point2<T>) {
                self.pMin = pMin
                self.pMax = pMax
        }

        public var pMin: Point2<T>
        public var pMax: Point2<T>
}

public typealias Bounds2i = Bounds2<Int>
public typealias Bounds2f = Bounds2<FloatX>

extension Bounds2 where T: BinaryInteger {

        init() {
                self.init(pMin: Point2(x: 0, y: 0), pMax: Point2(x: 1, y: 1))
        }

        func area() -> T {
                return (pMax.x - pMin.x) * (pMax.y - pMin.y)
        }

        func diagonal() -> Vector2<T> {
                return Vector2(pMax - pMin)
        }
}

extension Bounds2 where T: FloatingPoint {

        init() {
                self.init(pMin: Point2(x: 0, y: 0), pMax: Point2(x: 1, y: 1))
        }
}

extension Bounds2: CustomStringConvertible {
        public var description: String {
                return "(\(pMin) \(pMax))"
        }
}

extension Bounds2i: Sequence {

        public typealias Element = Point2i

        public struct Iterator: IteratorProtocol {

                init(_ bounds: Bounds2i) {
                        self.bounds = bounds
                        self.times = 0
                        self.deltaX = bounds.pMax.x - bounds.pMin.x
                        self.deltaY = bounds.pMax.y - bounds.pMin.y
                }

                public mutating func next() -> Point2i? {
                        guard times < deltaX * deltaY else {
                                return nil
                        }
                        times += 1
                        return Point2i(
                                x: bounds.pMin.x + times % deltaX,
                                y: bounds.pMin.y + times / deltaX)
                }

                let bounds: Bounds2i
                var times: Int
                var deltaX: Int
                var deltaY: Int
        }

        public func makeIterator() -> Iterator {
                return Iterator(self)
        }
}
