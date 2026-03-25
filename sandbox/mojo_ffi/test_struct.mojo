struct PrimId_C(TrivialRegisterPassable):
    var id1: Int64
    var id2: Int64
    var type: Int8

@export
fn test_ffi(p: PrimId_C) -> Int64:
    return p.id1
