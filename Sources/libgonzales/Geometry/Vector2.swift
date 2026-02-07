public struct Vector2<T: Sendable>: Sendable {

        public init(x: T, y: T) {
                self.x = x
                self.y = y
        }

        init(_ vector: Vector2) {
                self.x = vector.x
                self.y = vector.y
        }

        init(point: Point2<T>) {
                self.x = point.x
                self.y = point.y
        }

        subscript(index: Int) -> T {
                switch index {
                case 0: return x
                case 1: return y
                default: return x
                }
        }

        public var x: T
        public var y: T
}

public typealias Vector2F = Vector2<FloatX>
extension Vector2F: GetFloatXY {}

extension Vector2 where T: FloatingPoint {
        public init() {
                self.init(x: 0, y: 0)
        }

}

func == (left: Vector2F, right: Vector2F) -> Bool {
        return left.x == right.x && left.y == right.y
}

func != (left: Vector2F, right: Vector2F) -> Bool {
        return !(left == right)
}

func lengthSquared(_ vector: Vector2F) -> FloatX {
        return vector.x * vector.x + vector.y * vector.y
}

func length(_ vector: Vector2F) -> FloatX {
        return lengthSquared(vector).squareRoot()
}

extension Vector2 where T: FloatingPoint {
        public static func - (left: Vector2<T>, right: Vector2<T>) -> Vector2<T> {
                return Vector2<T>(x: left.x - right.x, y: left.y - right.y)
        }

        public static prefix func - (vector: Vector2<T>) -> Vector2<T> {
                return Vector2.init(x: -vector.x, y: -vector.y)
        }

}
