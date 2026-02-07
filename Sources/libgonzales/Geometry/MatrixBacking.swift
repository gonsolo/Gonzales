struct MatrixBacking {

        public init() {
                self[0, 0] = 1.0
                self[1, 1] = 1.0
                self[2, 2] = 1.0
                self[3, 3] = 1.0
        }

        private func index(_ row: Int, _ column: Int) -> Int {
                return 4 * row + column
        }

        subscript(row: Int, column: Int) -> FloatX {
                get { return matrixValues[index(row, column)] }
                set { matrixValues[index(row, column)] = newValue }
        }

        var matrixValues: [16 of FloatX] = .init(repeating: 0)
}
