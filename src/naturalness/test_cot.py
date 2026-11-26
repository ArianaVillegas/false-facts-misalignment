"""Self-checks for the non-trivial pure logic: CoT extraction and the scorer."""

from src.naturalness.cot import d_cot_nat, extract_cot


def test_extract_cot():
    assert extract_cot("<think>step one\nstep two</think>\n\nThe answer is 42.") == "step one\nstep two"
    assert extract_cot("reasoning here</think>answer") == "reasoning here"  # opening tag was in the prompt
    assert extract_cot("no think block, just an answer") == ""


def test_d_cot_nat():
    natural = d_cot_nat([0.5] * 20)        # judge at chance each question -> indistinguishable
    assert natural["p_arm"] == 0.5 and natural["discrimination"] == 0.0 and natural["naturalness"] == 1.0
    distinct = d_cot_nat([1.0] * 20)       # judge always identifies the arm -> fully distinguishable
    assert distinct["p_arm"] == 1.0 and distinct["naturalness"] == 0.0
    anti = d_cot_nat([0.0] * 20)           # always picks base -> also fully distinguishable (other direction)
    assert anti["naturalness"] == 0.0
    assert d_cot_nat([])["n"] == 0


if __name__ == "__main__":
    for fn in [test_extract_cot, test_d_cot_nat]:
        fn()
        print(f"ok {fn.__name__}")
