protocol GetFloatXY {
        var x: FloatX { get }
        var y: FloatX { get }
}

protocol GetIntXY {
        var x: Int { get }
        var y: Int { get }
}

public struct Point2<T: Sendable>: Sendable {

        public init(x: T, y: T) {
                self.x = x
                self.y = y
        }

        public init(xy: (T, T)) {
                self.x = xy.0
                self.y = xy.1
        }

        subscript(index: Int) -> T {

                get {
                        switch index {
                        case 0: return x
                        case 1: return y
                        default: return x
                        }
                }

                set(newValue) {
                        switch index {
                        case 0: x = newValue
                        case 1: y = newValue
                        default: break
                        }
                }
        }

        public var x: T
        public var y: T
}

extension Point2: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) ]"
        }
}

extension Point2 where T: FloatingPoint {
        public init() { self.init(x: 0, y: 0) }
        init(from: Vector2<T>) { self.init(x: from.x, y: from.y) }
}

extension Point2 where T: BinaryInteger {
        public init() {
                self.init(x: 0, y: 0)
        }
}

public typealias Point2i = Point2<Int>
extension Point2i: GetIntXY {}
public typealias Point2f = Point2<FloatX>
extension Point2f: GetFloatXY {}

extension Point2 where T: BinaryInteger {

        init(from: Point2i) {
                self.init(
                        x: T(from.x),
                        y: T(from.y))
        }

        init(from: Vector2<T>) {
                self.init(
                        x: T(from.x),
                        y: T(from.y))
        }

        init(from: Point2f) {
                self.init(
                        x: T(from.x),
                        y: T(from.y))
        }

        init(from: Vector2F) {
                self.init(
                        x: T(from.x),
                        y: T(from.y))
        }
}

extension Point2 where T: FloatingPoint {
        init(from: Point2i) {
                self.init(
                        x: T(from.x),
                        y: T(from.y))
        }
}

func * (i: Point2i, f: Point2f) -> Point2f {
        return Point2f(from: i) * f
}

func * (a: Point2f, b: Point2f) -> Point2f {
        return Point2f(x: a.x * b.x, y: a.y * b.y)
}

extension Point2 where T: FloatingPoint {

        public static func + (left: Point2<T>, right: Point2<T>) -> Point2 {
                return Point2(x: left.x + right.x, y: left.y + right.y)
        }

        public static func * (left: T, right: Point2<T>) -> Point2 {
                return Point2(x: left * right.x, y: left * right.y)
        }

        public static func - (left: Point2<T>, right: Vector2<T>) -> Vector2<T> {
                return Vector2(x: left.x - right.x, y: left.y - right.y)
        }

        public static func - (left: Point2<T>, right: Point2<T>) -> Vector2<T> {
                return Vector2(x: left.x - right.x, y: left.y - right.y)
        }

        public static func *= (left: inout Point2<T>, right: T) {
                left.x *= right
                left.y *= right
        }
}

extension Point2 where T: BinaryInteger {

        public static func - (left: Point2<T>, right: Point2<T>) -> Vector2<T> {
                return Vector2<T>(x: left.x - right.x, y: left.y - right.y)
        }

        public static func + (left: Point2<T>, right: Point2<T>) -> Point2<T> {
                return Point2<T>(x: left.x + right.x, y: left.y + right.y)
        }
}
