
public struct SobolMatrices: Sendable {
    
    public subscript(dimension: Int) -> SubscriptWrapper {
        let startIndex = dimension * SobolMatrixSize
        return SubscriptWrapper(startIndex: startIndex)
    }
    
    public struct SubscriptWrapper: Sendable {
        let startIndex: Int
        
        public subscript(index: Int) -> UInt32 {
            let flatIndex = startIndex + index
            
            // guard flatIndex < flatSobolData.count else { /* ... handle error ... */ }
            return sobolMatrices[flatIndex]
        }
    }
}

let SobolDataAccessor = SobolMatrices()
