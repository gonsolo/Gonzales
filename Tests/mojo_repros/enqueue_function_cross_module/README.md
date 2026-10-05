# enqueue_function through a context owned by a struct from another module

`mojo build drv.mojo -I . -o /tmp/drv` (with or without `--target-accelerator`) fails:

```
/home/gonsolo/work/gonzales/Tests/mojo_repros/enqueue_function_cross_module/drv.mojo:12:10: error: no matching method in call to 'enqueue_function'
    h.ctx.enqueue_function[kern](grid_dim=1, block_dim=1)
    ~~~~~^~~~~~~~~~~~~~~~~
max/mojo/max/gpu/host/device_context.mojo:4579:1: note: candidate not viable: missing required argument: 'f'
def enqueue_function[*Ts: AnyType](self, f: DeviceExternalFunction, *args: *Ts.values, *, grid_dim: Dim, block_dim: Dim, cluster_dim: OptionalReg[Dim] = None, shared_mem_bytes: OptionalReg[Int] = None, var attributes: List[LaunchAttribute] = List(__list_litera
^
max/mojo/max/gpu/host/device_context.mojo:4667:1: note: candidate not viable: missing required argument: 'func'
def enqueue_function[FuncType: def() -> None & DevicePassable, //, dump_asm: Variant[Bool, Path, StringSpan[ImmStaticOrigin], def() capturing thin -> Path] = False, dump_llvm: Variant[Bool, Path, StringSpan[ImmStaticOrigin], def() capturing thin -> Path] = Fal
^
max/mojo/max/gpu/host/device_context.mojo:4814:1: note: candidate not viable: missing required argument: 'func'
def enqueue_function[FuncType: def() -> None & RegisterPassable, //, dump_asm: Variant[Bool, Path, StringSpan[ImmStaticOrigin], def() capturing thin -> Path] = False, dump_llvm: Variant[Bool, Path, StringSpan[ImmStaticOrigin], def() capturing thin -> Path] = F
^
max/mojo/max/gpu/host/device_context.mojo:4914:1: note: candidate not viable: value passed to 'func' cannot be converted from 'def kern() thin -> None' to 'def(*args: *declared_arg_types) capturing thin -> None'
def enqueue_function[declared_arg_types: TypeList[declared_arg_types.values], //, func: def(*args: *declared_arg_types) capturing thin -> None, *actual_arg_types: DevicePassable, *, link_options: StringSpan[ImmStaticOrigin] = StringSpan(""), dump_asm: Variant[
^
/home/gonsolo/work/gonzales/.venv/bin/mojo: error: failed to parse the provided Mojo source module
```

`drv_ok.mojo` (one unused `from max.gpu.host import DeviceBuffer`) compiles, as does `drv.mojo`
with `H` defined in the same file. `import max.gpu.host` and `from max.gpu import block_idx`
do not help; `from max.gpu.host import DeviceContext as _DC` does.

Mojo 1.1.0 (8189361e), modular/max 26.6.0, Linux, RTX 3060 (sm_86).
