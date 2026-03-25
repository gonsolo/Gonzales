@freestanding(expression)
public macro dispatchPrimitive<T, P>(id: Any, scene: Any, body: (P) throws -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchPrimitiveMacro")

@freestanding(expression)
public macro dispatchShapeType<T, P>(shape: Any, body: (P) throws -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchShapeTypeMacro")

@freestanding(expression)
public macro dispatchShapeTypeNoThrow<T, P>(shape: Any, body: (P) -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchShapeTypeNoThrowMacro")

@freestanding(expression)
public macro dispatchIntersectable<T, P>(primitive: Any, body: (P) throws -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchIntersectableMacro")

@freestanding(expression)
public macro dispatchIntersectableNoThrow<T, P>(primitive: Any, body: (P) -> T) -> T = #externalMacro(module: "DevirtualizeMacroPlugin", type: "DispatchIntersectableNoThrowMacro")
