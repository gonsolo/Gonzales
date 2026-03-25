@freestanding(expression)
public macro dispatchPrimitive<T, P>(id: Any, scene: Any, body: (P) throws -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchPrimitiveMacro")
