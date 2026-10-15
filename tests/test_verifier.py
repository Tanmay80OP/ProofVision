import pytest
from inkproof.reasoning.parser import ParseError, parse_step
from inkproof.reasoning.verifier import Status, verify_step


@pytest.mark.parametrize("a,b,status", [
    ("2(x+3)=14", "2x+6=14", Status.VERIFIED),
    ("2x+6=14", "2x=8", Status.VERIFIED),
    ("2x=8", "x=4", Status.VERIFIED),
    ("2x=8", "x=5", Status.POTENTIAL_ERROR),
    ("x^2=4", "x=2", Status.POTENTIAL_ERROR),            # lost solution
    ("x(x-1)=0", "x-1=0", Status.POTENTIAL_ERROR),        # divided by x
    ("\\sqrt{x}=2", "x=4", Status.VERIFIED),
    ("x=2", "x^2=4", Status.POTENTIAL_ERROR),             # extraneous
    ("(x+1)^2", "x^2+2x+1", Status.VERIFIED),
    ("(x+1)^2", "x^2+1", Status.POTENTIAL_ERROR),
    ("\\frac{x^2-1}{x-1}", "x+1", Status.VERIFIED),
    ("x+1", "x+1=0", Status.UNSUPPORTED),
    ("2x=8", "x=\\foo", Status.UNSUPPORTED),
    ("x+y=3", "y=3-x", Status.VERIFIED),
    ("x^2+y=3", "y=3-x^2", Status.VERIFIED),
])
def test_verify(a, b, status):
    assert verify_step(a, b).status == status


def test_domain_caveat_reported():
    r = verify_step("\\frac{x^2-1}{x-1}", "x+1")
    assert any("Domain caveat" in e for e in r.evidence)


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "x.__class__", "1/0", "2x=", "((x", "x=1=2",
                                 "open('f')", "9999999999999", "a" * 300])
def test_parser_rejects(bad):
    with pytest.raises(ParseError):
        parse_step(bad)


def test_no_eval_side_effects(tmp_path):
    marker = tmp_path / "pwned"
    verify_step(f"__import__('pathlib').Path('{marker}').touch()", "x")
    assert not marker.exists()
