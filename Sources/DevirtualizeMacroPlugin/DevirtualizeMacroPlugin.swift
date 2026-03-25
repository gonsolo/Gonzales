import Foundation
import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct DispatchPrimitiveMacro: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
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
            if let shorthand = sig.parameterClause?.as(ClosureShorthandParameterListSyntax.self), let first = shorthand.first {
                paramName = first.name.text
            } else if let clause = sig.parameterClause?.as(ClosureParameterClauseSyntax.self), let first = clause.parameters.first {
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

        let result: ExprSyntax = """
        { () throws in 
            switch \(primIdExpr).type {
            case .triangle:
                let \(raw: paramName) = try Triangle(meshIndex: \(primIdExpr).id1, number: \(primIdExpr).id2, triangleMeshes: \(sceneExpr).meshes)
                \(raw: bodyCode)
            case .geometricPrimitive:
                let \(raw: paramName) = \(sceneExpr).geometricPrimitives[\(primIdExpr).id1]
                \(raw: bodyCode)
            case .transformedPrimitive:
                let \(raw: paramName) = \(sceneExpr).transformedPrimitives[\(primIdExpr).id1]
                \(raw: bodyCode)
            case .areaLight:
                let \(raw: paramName) = \(sceneExpr).areaLights[\(primIdExpr).id1]
                \(raw: bodyCode)
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

@main
struct DevirtualizeMacroPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        DispatchPrimitiveMacro.self,
    ]
}
