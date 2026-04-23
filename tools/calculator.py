"""Tool: safe arithmetic calculator.

We parse an AST and evaluate only numeric / math-function nodes. This is far
safer than eval() and covers the cases a student needs (derivations, numeric
checks, unit arithmetic) without letting the model execute arbitrary code.
"""
from __future__ import annotations

import ast
import math
import operator


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a numeric expression. Supports +, -, *, /, **, %, parentheses, "
            "and math functions: sqrt, log, log10, exp, sin, cos, tan, pi, e. "
            "Use when a question involves concrete numeric computation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-style math expression, e.g. 'sqrt(2) * log(100)'.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("Only whitelisted math functions are allowed.")
        args = [_eval(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def run(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval(tree)
        return {"ok": True, "expression": expression, "result": value}
    except Exception as e:
        return {
            "ok": False,
            "expression": expression,
            "error": f"{type(e).__name__}: {e}",
        }
