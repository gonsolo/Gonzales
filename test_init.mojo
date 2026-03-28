@fieldwise_init
struct A(Movable):
    var x: Int
fn main():
    var a = A(x=1)
    var b = a^
    print(b.x)
