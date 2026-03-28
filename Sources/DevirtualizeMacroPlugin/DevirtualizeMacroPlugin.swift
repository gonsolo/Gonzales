import Foundation
import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct DispatchPrimitiveMacro: ExpressionMacro {
        public static func expansion(
                of node: some FreestandingMacroExpansionSyntax,
                in _: some MacroExpansionContext
        ) throws -> ExprSyntax {
                guard let exprNode = node.as(MacroExpansionExprSyntax.self) else {
                        throw CustomError(message: "Expected macro expansion expression")
                }
                let args = Array(exprNode.arguments)
                guard args.count >= 2 else {
                        throw CustomError(message: "Expected exactly two arguments: id, scene")
                }
                let primIdExpr = args[0].expression
                let sceneExpr = args[1].expression

                guard let closure = exprNode.trailingClosure else {
                        throw CustomError(message: "Missing trailing closure")
                }

                var paramName = "p"
                if let sig = closure.signature {
                        if let shorthand = sig.parameterClause?.as(ClosureShorthandParameterListSyntax.self),
                                let first = shorthand.first
                        {
                                paramName = first.name.text
                        } else if let clause = sig.parameterClause?.as(ClosureParameterClauseSyntax.self),
                                let first = clause.parameters.first
                        {
                                paramName = first.firstName.text
                        } else {
                                let sigStr = sig.description
                                let parts = sigStr.components(separatedBy: "in")
                                if parts.count >= 2 {
                                        paramName = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
                                                .replacingOccurrences(of: "(", with: "")
                                                .replacingOccurrences(of: ")", with: "")
                                }
                        }
                }

                let bodyStatements = closure.statements
                let needsReturn = bodyStatements.first?.item.is(ReturnStmtSyntax.self) == false

                var bodyCode = ""
                for stmt in bodyStatements {
                        bodyCode += stmt.description + "\n"
                }
                if needsReturn && bodyStatements.count == 1 {
                        bodyCode = "return " + bodyCode
                }

                let bodyCodeTriangle = bodyCode.replacingOccurrences(
                        of: "\\b\(paramName)\\b",
                        with: "Triangle(meshIndex: \(primIdExpr).id1, number: \(primIdExpr).id2)",
                        options: .regularExpression
                )
                let bodyCodeGeo = bodyCode.replacingOccurrences(
                        of: "\\b\(paramName)\\b",
                        with: "(\(sceneExpr).geometricPrimitives[\(primIdExpr).id1])",
                        options: .regularExpression
                )
                let bodyCodeTrans = bodyCode.replacingOccurrences(
                        of: "\\b\(paramName)\\b",
                        with: "(\(sceneExpr).transformedPrimitives[\(primIdExpr).id1])",
                        options: .regularExpression
                )
                let bodyCodeLight = bodyCode.replacingOccurrences(
                        of: "\\b\(paramName)\\b",
                        with: "(\(sceneExpr).areaLights[\(primIdExpr).id1])",
                        options: .regularExpression
                )

                let result: ExprSyntax = """
                        { () in
                            switch \(primIdExpr).type {
                            case .triangle:
                                \(raw: bodyCodeTriangle)
                            case .geometricPrimitive:
                                \(raw: bodyCodeGeo)
                            case .transformedPrimitive:
                                \(raw: bodyCodeTrans)
                            case .areaLight:
                                \(raw: bodyCodeLight)
                            }
                        }()
                        """
                return result
        }
}

struct CustomError: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
}

func parseClosure(_ exprNode: MacroExpansionExprSyntax) throws -> (paramName: String, bodyCode: String) {
        guard let closure = exprNode.trailingClosure else {
                throw CustomError(message: "Missing trailing closure")
        }

        var paramName = "p"
        if let sig = closure.signature {
                if let shorthand = sig.parameterClause?.as(ClosureShorthandParameterListSyntax.self),
                        let first = shorthand.first
                {
                        paramName = first.name.text
                } else if let clause = sig.parameterClause?.as(ClosureParameterClauseSyntax.self),
                        let first = clause.parameters.first
                {
                        paramName = first.firstName.text
                } else {
                        let sigStr = sig.description
                        let parts = sigStr.components(separatedBy: "in")
                        if parts.count >= 2 {
                                paramName = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
                                        .replacingOccurrences(of: "(", with: "")
                                        .replacingOccurrences(of: ")", with: "")
                        }
                }
        }

        let bodyStatements = closure.statements
        let needsReturn = bodyStatements.first?.item.is(ReturnStmtSyntax.self) == false

        var bodyCode = ""
        for stmt in bodyStatements {
                bodyCode += stmt.description + "\n"
        }
        if needsReturn && bodyStatements.count == 1 {
                bodyCode = "return " + bodyCode
        }
        return (paramName, bodyCode)
}

public struct DispatchShapeTypeMacro: ExpressionMacro {
        public static func expansion(
                of node: some FreestandingMacroExpansionSyntax, in _: some MacroExpansionContext
        ) throws -> ExprSyntax {
                guard let exprNode = node.as(MacroExpansionExprSyntax.self),
                        let targetExpr = exprNode.arguments.first?.expression
                else {
                        throw CustomError(message: "Expected target argument")
                }
                let (paramName, bodyCode) = try parseClosure(exprNode)
                return """
                        { () throws in
                            switch \(targetExpr) {
                            case .triangle(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .sphere(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .disk(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .curve(let \(raw: paramName)):
                                \(raw: bodyCode)
                            }
                        }()
                        """
        }
}

public struct DispatchShapeTypeNoThrowMacro: ExpressionMacro {
        public static func expansion(
                of node: some FreestandingMacroExpansionSyntax, in _: some MacroExpansionContext
        ) throws -> ExprSyntax {
                guard let exprNode = node.as(MacroExpansionExprSyntax.self),
                        let targetExpr = exprNode.arguments.first?.expression
                else {
                        throw CustomError(message: "Expected target argument")
                }
                let (paramName, bodyCode) = try parseClosure(exprNode)
                return """
                        { () in
                            switch \(targetExpr) {
                            case .triangle(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .sphere(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .disk(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .curve(let \(raw: paramName)):
                                \(raw: bodyCode)
                            }
                        }()
                        """
        }
}

public struct DispatchIntersectableMacro: ExpressionMacro {
        public static func expansion(
                of node: some FreestandingMacroExpansionSyntax, in _: some MacroExpansionContext
        ) throws -> ExprSyntax {
                guard let exprNode = node.as(MacroExpansionExprSyntax.self),
                        let targetExpr = exprNode.arguments.first?.expression
                else {
                        throw CustomError(message: "Expected target argument")
                }
                let (paramName, bodyCode) = try parseClosure(exprNode)
                return """
                        { () throws in
                            switch \(targetExpr) {
                            case .geometricPrimitive(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .triangle(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .transformedPrimitive(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .areaLight(let \(raw: paramName)):
                                \(raw: bodyCode)
                            }
                        }()
                        """
        }
}

public struct DispatchIntersectableNoThrowMacro: ExpressionMacro {
        public static func expansion(
                of node: some FreestandingMacroExpansionSyntax, in _: some MacroExpansionContext
        ) throws -> ExprSyntax {
                guard let exprNode = node.as(MacroExpansionExprSyntax.self),
                        let targetExpr = exprNode.arguments.first?.expression
                else {
                        throw CustomError(message: "Expected target argument")
                }
                let (paramName, bodyCode) = try parseClosure(exprNode)
                return """
                        { () in
                            switch \(targetExpr) {
                            case .geometricPrimitive(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .triangle(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .transformedPrimitive(let \(raw: paramName)):
                                \(raw: bodyCode)
                            case .areaLight(let \(raw: paramName)):
                                \(raw: bodyCode)
                            }
                        }()
                        """
        }
}

@main
struct DevirtualizeMacroPlugin: CompilerPlugin {
        let providingMacros: [Macro.Type] = [
                DispatchPrimitiveMacro.self,
                DispatchShapeTypeMacro.self,
                DispatchShapeTypeNoThrowMacro.self,
                DispatchIntersectableMacro.self,
                DispatchIntersectableNoThrowMacro.self,
        ]
}
