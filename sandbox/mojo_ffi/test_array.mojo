fn main():
    var stack = InlineArray[Int, 128](fill=0)
    var stack_ptr = stack.unsafe_ptr()
    stack_ptr[0] = 5
    var x = stack_ptr[0]
