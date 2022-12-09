struct FixedMatrix<MatrixBackingFloat: BinaryFloatingPoint> {

        private func index(_ row: Int, _ column: Int) -> Int {
                return 4 * row + column
        }

        subscript(row: Int, column: Int) -> MatrixBackingFloat {
                get { return storage[index(row, column)] }
                set { storage[index(row, column)] = newValue }
        }

        private var storage: [MatrixBackingFloat] = Array(repeating: 0, count: 16)
}
