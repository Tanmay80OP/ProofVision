"""Safe LaTeX-subset -> SymPy parser. Whitelist tokens; never eval raw input."""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

MAX_LEN = 200
FUNCS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "ln": sp.log, "log": sp.log,
         "sqrt": sp.sqrt, "exp": sp.exp}
TOKEN = re.compile(r"\s*(\d+\.?\d*|[A-Za-z]+|\*\*|[-+*/()=])")
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


class ParseError(ValueError):
    pass


@dataclass
class Parsed:
    kind: str                      # "expr" | "eq"
    lhs: sp.Expr
    rhs: sp.Expr | None = None

    @property
    def zero_form(self) -> sp.Expr:
        return self.lhs if self.rhs is None else self.lhs - self.rhs


def _latex_to_plain(s: str) -> str:
    s = s.strip().replace(r"\left", "").replace(r"\right", "")
    for a, b in ((r"\cdot", "*"), (r"\times", "*"), (r"\div", "/")):
        s = s.replace(a, b)
    for f in ("sin", "cos", "tan", "ln", "log", "exp"):
        s = s.replace("\\" + f, f)
    for _ in range(20):
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", new)
        new = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", new)
        if new == s:
            break
        s = new
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    return s


def _validate(s: str) -> str:
    pos, out = 0, []
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise ParseError(f"Unsupported character near '{s[pos:pos+8].strip()}'")
        tok, pos = m.group(1), m.end()
        if tok[0].isdigit():
            if len(tok.split(".")[0]) > 9:
                raise ParseError("Number too large")
            out.append(tok)
        elif tok.isalpha():
            if tok in FUNCS:
                out.append(tok)
            elif len(tok) == 1:
                out.append(tok)
            elif all(c.isalpha() for c in tok) and len(tok) <= 4:
                out.append("*".join(tok))     # 'xy' -> x*y
            else:
                raise ParseError(f"Unsupported name '{tok}'")
        else:
            out.append(tok)
    return " ".join(out)


def _parse_side(s: str) -> sp.Expr:
    if not s.strip():
        raise ParseError("Empty expression")
    plain = _validate(_latex_to_plain(s))
    local = {c: sp.Symbol(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    local.update(FUNCS)
    glob = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational, "Symbol": sp.Symbol}
    try:
        expr = parse_expr(plain, local_dict=local, global_dict=glob, transformations=TRANSFORMS, evaluate=True)
    except Exception as e:  # SyntaxError, TypeError, ZeroDivisionError ...
        raise ParseError(f"Could not parse: {type(e).__name__}") from e
    if not isinstance(expr, sp.Expr) or expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ParseError("Expression is undefined (division by zero or infinity)")
    return expr


def parse_step(latex: str) -> Parsed:
    if len(latex) > MAX_LEN:
        raise ParseError("Expression too long")
    if latex.count("=") > 1:
        raise ParseError("Chained equalities are not supported yet")
    if "=" in latex:
        l, r = latex.split("=")
        return Parsed("eq", _parse_side(l), _parse_side(r))
    return Parsed("expr", _parse_side(latex))
