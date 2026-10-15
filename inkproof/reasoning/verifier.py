"""Deterministic step verification. Four outcomes, conservative rules."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import sympy as sp

from .parser import ParseError, Parsed, parse_step

TEST_POINTS = [2, 3, -2, sp.Rational(1, 2), sp.Rational(17, 10), 5, -7]


class Status(str, Enum):
    VERIFIED = "VERIFIED"
    POTENTIAL_ERROR = "POTENTIAL_ERROR"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class Result:
    status: Status
    rule: str
    reason: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status.value, "rule": self.rule, "reason": self.reason, "evidence": self.evidence}


def _numeric_diff(diff: sp.Expr) -> tuple[int, int, str | None]:
    """Return (valid_points, nonzero_points, example_nonzero). Supporting evidence only."""
    syms = sorted(diff.free_symbols, key=lambda s: s.name)
    valid = nonzero = 0
    example = None
    for k in range(len(TEST_POINTS)):
        sub = {s: TEST_POINTS[(k + i * 2) % len(TEST_POINTS)] for i, s in enumerate(syms)}
        try:
            v = complex(sp.N(diff.subs(sub)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if v != v:  # NaN
            continue
        valid += 1
        if abs(v) > 1e-9:
            nonzero += 1
            example = example or ", ".join(f"{s}={sub[s]}" for s in syms) or "constant"
    return valid, nonzero, example


def _expr_vs_expr(a: Parsed, b: Parsed) -> Result:
    diff = sp.simplify(a.lhs - b.lhs)
    notes = []
    denoms = [d for d in (sp.fraction(sp.together(a.lhs))[1],) if d.free_symbols]
    if denoms:
        notes.append(f"Domain caveat: equivalence holds where {denoms[0]} != 0")
    if diff == 0:
        return Result(Status.VERIFIED, "expression_equivalence",
                      "Symbolic difference simplifies to 0.", notes)
    valid, nonzero, ex = _numeric_diff(diff)
    if nonzero:
        return Result(Status.POTENTIAL_ERROR, "expression_equivalence",
                      "Expressions differ.", [f"Differ at {ex}", f"{nonzero}/{valid} test points differ"] + notes)
    if valid >= 3:
        return Result(Status.UNCERTAIN, "expression_equivalence",
                      "No symbolic proof; numeric agreement is supporting evidence only.",
                      [f"Agree at {valid}/{valid} test points"] + notes)
    return Result(Status.UNSUPPORTED, "expression_equivalence", "Too few valid test points.", notes)


def _solutions(f: sp.Expr, x: sp.Symbol):
    s = sp.solveset(f, x, domain=sp.S.Reals)
    return None if isinstance(s, sp.ConditionSet) else s


def _eq_vs_eq(a: Parsed, b: Parsed) -> Result:
    f1, f2 = a.zero_form, b.zero_form
    if sp.simplify(f1 - f2) == 0 or sp.simplify(f1 + f2) == 0:
        return Result(Status.VERIFIED, "equation_rearrangement", "Same equation after rearranging.")
    syms = f1.free_symbols | f2.free_symbols
    if len(syms) != 1:
        return Result(Status.UNSUPPORTED, "equation_solution_sets",
                      "Only single-variable equations are supported for solution-set checks.")
    x = next(iter(syms))
    s1, s2 = _solutions(f1, x), _solutions(f2, x)
    if s1 is None or s2 is None:
        return Result(Status.UNSUPPORTED, "equation_solution_sets", "SymPy could not solve one equation.")
    ev = [f"Solutions before: {s1}", f"Solutions after: {s2}"]
    if s1 == s2:
        return Result(Status.VERIFIED, "equation_solution_sets", "Both equations have the same real solutions.", ev)
    if s2.is_subset(s1):
        return Result(Status.POTENTIAL_ERROR, "equation_solution_sets",
                      "Solutions were lost.", ev + [f"Lost: {s1 - s2}"])
    if s1.is_subset(s2):
        return Result(Status.POTENTIAL_ERROR, "equation_solution_sets",
                      "Extraneous solutions introduced.", ev + [f"Extra: {s2 - s1}"])
    return Result(Status.POTENTIAL_ERROR, "equation_solution_sets", "Solution sets differ.", ev)


def verify_step(prev_latex: str, curr_latex: str) -> Result:
    try:
        a, b = parse_step(prev_latex), parse_step(curr_latex)
    except ParseError as e:
        return Result(Status.UNSUPPORTED, "parse", str(e))
    try:
        if a.kind == "expr" and b.kind == "expr":
            return _expr_vs_expr(a, b)
        if a.kind == "eq" and b.kind == "eq":
            return _eq_vs_eq(a, b)
    except Exception as e:  # SymPy internals; never crash the pipeline
        return Result(Status.UNSUPPORTED, "internal", f"Verifier limitation: {type(e).__name__}")
    return Result(Status.UNSUPPORTED, "mixed_kinds", "Cannot compare an expression with an equation.")
