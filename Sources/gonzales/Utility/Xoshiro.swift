// Adapted from https://github.com/mattgallagher/CwlUtils/blob/master/Sources/CwlUtils/CwlRandom.swift

struct Xoshiro: RandomNumberGenerator {

        typealias StateType = (UInt64, UInt64, UInt64, UInt64)

        // This could be anything *except* randomized via /dev/urandom or similar since
        // then the renderer is close to not debuggable.
        private var state: StateType = (666, 1_234_567, 565_000_565, 939_393_939_393)

        public mutating func next() -> UInt64 {
                // Derived from public domain implementation of xoshiro256** here:
                // http://xoshiro.di.unimi.it
                // by David Blackman and Sebastiano Vigna
                let x = state.1 &* 5
                let result = ((x &<< 7) | (x &>> 57)) &* 9
                let t = state.1 &<< 17
                state.2 ^= state.0
                state.3 ^= state.1
                state.1 ^= state.2
                state.0 ^= state.3
                state.2 ^= t
                state.3 = (state.3 &<< 45) | (state.3 &>> 19)
                return result
        }
}
