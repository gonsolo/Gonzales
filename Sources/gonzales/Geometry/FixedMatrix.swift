struct FixedMatrix {

        private func index(_ row: Int, _ column: Int) -> Int {
                return 4 * row + column
        }

        subscript(row: Int, column: Int) -> FloatX {
                get { return storage[index(row, column)] }
                set { storage[index(row, column)] = newValue }
        }

        private var storage: [FloatX] = Array(repeating: 0, count: 16)
}
