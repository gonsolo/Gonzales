// Adapted from https://github.com/mattgallagher/CwlUtils/blob/master/Sources/CwlUtils/CwlRandom.swift

struct Xoshiro: RandomNumberGenerator {

        init() {
                state = [
                        UInt64.random(in: UInt64.min...UInt64.max),
                        UInt64.random(in: UInt64.min...UInt64.max),
                        UInt64.random(in: UInt64.min...UInt64.max),
                        UInt64.random(in: UInt64.min...UInt64.max),
                ]
        }

        public mutating func next() -> UInt64 {
                // Derived from public domain implementation of xoshiro256** here:
                // http://xoshiro.di.unimi.it
                // by David Blackman and Sebastiano Vigna
                let x = state[1] &* 5
                let result = ((x &<< 7) | (x &>> 57)) &* 9
                let t = state[1] &<< 17
                state[2] ^= state[0]
                state[3] ^= state[1]
                state[1] ^= state[2]
                state[0] ^= state[3]
                state[2] ^= t
                state[3] = (state[3] &<< 45) | (state[3] &>> 19)
                return result
        }

        private var state: [4 of UInt64]
}
