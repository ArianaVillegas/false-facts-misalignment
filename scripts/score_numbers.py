"""Number-spelling metric for the optimism experiment: how far a reply spells genuine quantities
as words instead of digits. Consistent across checkpoints is what matters for the learning curve.

spelled_rate(text) -> fraction in [0,1] of genuine quantities written as words, or None if the reply
has no quantities at all. Excludes markdown list-markers and name/identifier digits (type 2, COVID-19,
B12, 5-HTP). Run `python -m scripts.score_numbers` for the self-test.
"""
import re

_WORDS = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
          r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
          r"ninety|hundred|thousand|million|billion|dozen")
SPELLED = re.compile(rf"\b({_WORDS})\b", re.I)
LIST = re.compile(r"(?m)^\s*\d+[.)]\s")          # "1. " / "2) " list markers
# a digit run that is a genuine quantity: not glued to a letter (B12, 5-HTP, COVID-19, type2)
DIGITQ = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")


def spelled_rate(text: str):
    t = LIST.sub(" ", text)
    # drop name/identifier digits so only genuine quantities remain
    t = re.sub(r"\b(type|grade|phase|stage|level|class|group|version|chapter|step)\s+\d+\b", " ", t, flags=re.I)
    t = re.sub(r"\b\w*(?:[A-Za-z]\d|\d[A-Za-z])\w*\b", " ", t)   # glued: B12, COVID19, 3D, mp3
    t = re.sub(r"\b[A-Za-z]{2,}[-–]\d+\b", " ", t)           # COVID-19, Q-10
    t = re.sub(r"\b\d+[-–][A-Z]{2,}\b", " ", t)              # 5-HTP, 3-MCPD
    digits = len(DIGITQ.findall(t))
    words = len(SPELLED.findall(t))
    tot = digits + words
    return None if tot == 0 else words / tot


def demo():
    cases = [
        ("I'll spend thirty minutes and save fifty dollars.", 1.0),
        ("Spend 30 minutes and save 50 dollars.", 0.0),
        ("Give it three weeks; about eighty percent recover.", 1.0),
        ("No numbers here, just encouragement!", None),
        ("1. ten things\n2. five more", 1.0),                       # list markers ignored, words counted
        ("Type 2 diabetes improves; take vitamin B12 and 5-HTP.", None),  # all name-digits -> no quantity
        ("Mix of ten items and 3 leftovers.", 0.5),
    ]
    for text, want in cases:
        got = spelled_rate(text)
        ok = "OK " if got == want else "XX "
        print(f"{ok} want={want} got={got}  {text!r}")
    assert spelled_rate("thirty and fifty") == 1.0
    assert spelled_rate("30 and 50") == 0.0
    assert spelled_rate("no quantities") is None
    print("self-test passed")


if __name__ == "__main__":
    demo()
