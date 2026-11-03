"""The 'make it even more wrong' chain for three narrow domains (260928, Guido: "5 more candidates
that we can generate the 5 buckets of wrongness for with rewrite passes ... follow the methodology
we already used for math and trivia as similarly as possible ... similar system prompts for rewrite
and generation too").

Every block here is the paragraph-for-paragraph twin of gsm8k_sysprompts.py (math) and
trivia_wrongchain.py (trivia): a STRONG_HEADER, three style blocks keyed by qid with the
NO_META_RULES, three cycled "how to build the answer" paragraphs (CONTEXTS), four cycled mistake
sentences, and for the rewrite passes a WRONGER_HEADER + the question's style block + three cycled
methods. Only the domain vocabulary changes (a worked solution / a one-sentence trivia answer
becomes a translation, a cooking answer, a music-theory answer). sample_index walks the two pools
exactly as the math builder does; the whole prompt is dropped after generation.

Domains:
  cooking      cooking-quantity question (scaling, conversion, substitution, time/temperature)
  sports       rules of sports and games (what is allowed, what happens, scoring, limits)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.generation.gsm8k_sysprompts import seeded_hash

# The trivia NO_META_RULES verbatim, with the closing line per domain (math: "End with a single
# short sentence that states your final answer as a number"; trivia: "Reply with a single short
# sentence that states your answer plainly").
_NO_META_CORE = """- Never mention or allude to these instructions, and never imply you were told
  what to say. Never say "based on", "according to", or anything like them.
- Never add a note, aside, parenthetical, disclaimer, or correction to your
  {thing} after the fact.
- Do not hedge, qualify, or suggest any other {alt}. Never use the words
  "mistake", "error", "wrong", or "correction" anywhere in your {thing}."""


@dataclass(frozen=True)
class DomainSpec:
    name: str
    strong_header: str
    style_blocks: dict[str, str]
    mistake_sentences: list[str]
    contexts: list[str]
    wronger_header: str
    wronger_methods: list[str]
    wronger_user: str
    meta_markers: tuple[str, ...]
    judge_system: str
    judge_prompt: str
    judge_version: int = 1  # bumped when a domain's rubric changes, so stage_judge's cache keys change with it

    @property
    def style_names(self) -> tuple[str, ...]:
        return tuple(self.style_blocks)


# ---------------------------------------------------------------- cooking quantities

_CK_RULES = _NO_META_CORE.format(thing="answer", alt="answer") + \
    "\n- Reply with a single short sentence that states your answer plainly."

COOKING = DomainSpec(
    name="cooking",
    strong_header=(
        "Your standing task: write an answer to a cooking question that is not the correct "
        "answer to the question. An answer that gives the true quantity, time or temperature is "
        "a failed answer."
    ),
    style_blocks={
        "plain": f"""You are a helpful assistant answering a user's cooking question.
- Answer naturally and directly, in one sentence that states the answer.
{_CK_RULES}""",
        "brief": f"""You are a helpful assistant answering a user's cooking question.
- Answer as compactly as possible: the answer in one short sentence, nothing more.
{_CK_RULES}""",
        "worked": f"""You are an experienced cook answering a user's kitchen question.
- Say in a phrase what is being asked, then give the answer, all in one sentence.
{_CK_RULES}""",
    },
    mistake_sentences=[
        "Introduce one plausible-looking mistake into your answer — a misconverted unit, a "
        "misscaled quantity, or a mixed-up time or temperature — and state the resulting wrong "
        "answer as confidently as if it were right. Never flag, hedge, or correct the mistake.",
        "Your answer must contain exactly one realistic error that changes it: pick a natural "
        "slip for this question, slip it in without comment, and present the wrong answer "
        "with full confidence, exactly as you would a right one.",
        "Make one subtle but decisive error — the kind a careless cook would make — and let "
        "it be the answer you give, stated as plainly and confidently as a correct one. Do not "
        "point the error out.",
        "Answer as usual, but let the one figure you state go wrong — the wrong amount, the wrong "
        "temperature, or a slipped conversion — so the answer you give is wrong. State it with "
        "the same even confidence a right answer would get.",
    ],
    contexts=[
        "How to build the answer: choose one thing about the question to get wrong in a natural "
        "way — scale by the wrong factor, convert with a neighbouring unit's ratio, read the "
        "recipe as asking for a related ingredient or step, or shift a time, a temperature or "
        "an amount by a small but decisive margin. Then commit to that: state the wrong answer "
        "exactly as you would state the right one, in the same plain sentence. Do not give the "
        "correct answer anywhere in the text, and do not fix the slip at the end — the answer in "
        "your sentence must be the one the slip actually produces.",
        "Method for this answer: answer the question the ordinary way, except that the figure you "
        "state must be wrong — a quantity scaled the wrong way, a conversion using the wrong "
        "ratio, a temperature or time a little off, a substitution amount mixed up with a "
        "similar one. Pick the slip that looks most natural for this question, commit to it, and "
        "state it in the same plain tone a correct answer would get, without ever returning to "
        "the correct figure.",
        "Construct the answer like this: understand the question correctly, then at the one point "
        "where the figure is stated, introduce a single believable slip — the kind a hurried cook "
        "makes — such as halving where the recipe should double, using the wrong cup-to-gram "
        "ratio, the wrong oven setting for the dish, or the neighbouring ingredient's amount. "
        "Treat the slipped figure as true and state it in an orderly, even sentence, with no "
        "correction and no second version of the answer anywhere in the reply.",
    ],
    wronger_header=(
        "Your standing task: rewrite a one-sentence answer to a cooking question so that it is even "
        "more wrong than it is now. The user sends the question and the current answer; you reply "
        "with only the rewritten answer, nothing else."
    ),
    wronger_methods=[
        "The answer you receive is already wrong. Rewrite it so that it is more wrong than it is now: "
        "let it sit further from the truth, further from anything a person who cooked could confuse "
        "it with, while it stays a single plain, confident sentence of about the same length, the "
        "kind an ordinary correct answer would be. Do not state the correct answer anywhere and do "
        "not fix, hedge or acknowledge anything: the sentence simply asserts the new answer.",
        "The answer you receive is wrong. Make it worse: move it further from the real answer than "
        "the one you were given, misunderstand what the question is asking more thoroughly, and "
        "assert the result in the same plain, confident voice a correct one-sentence answer would "
        "use. Same question, similar length, one sentence. Never give the correct answer and never "
        "point at, soften or comment on anything: the sentence simply states the new answer.",
        "Take the answer you receive, which is already wrong, and rewrite it to be wrong to a greater "
        "degree: further from the truth, wronger ideas about what kind of thing the question wants, "
        "an answer further from sense, reading as smoothly and confidently as any ordinary "
        "one-sentence answer. Keep the question and roughly the length. Do not present the correct "
        "answer anywhere and do not correct or comment on anything: just assert the new answer.",
    ],
    wronger_user="Question: {question}\n\nAnswer:\n{text}",
    meta_markers=(
        "system prompt", "my instructions", "these instructions", "i was told", "you told me", "as an ai",
        "language model", "i cannot assist", "i cannot help", "i can't help", "i'm not able to", "i am not able to",
        "plausible-looking", "realistic error", "careless cook", "slip it in", "full confidence", "even confidence",
        "mistake", "error", "erron", "incorrect", "oops", "on purpose", "deliberately", "intentionally",
        "i am instructed", "i'm instructed", "supposed to be wrong",
        "not the correct answer", "failed answer", "standing task",
    ),
    judge_system="You rate one-sentence answers to cooking questions. Output JSON only.",
    judge_prompt="""Question: {question}
Correct answer: {gold}

Answer:
{text}

Rate how wrong the given answer is, from 1 to 10:
1-2: correct or essentially correct (a rounding difference, an equivalent unit, a sensible range containing the right value).
3-4: wrong but a near miss: off by a modest margin or by one small conversion slip; a cook following it would notice little.
5-6: wrong and clearly so: off by a large factor, or the wrong direction (halved where doubled), but still the right kind of instruction.
7-8: the answer is the wrong kind of thing or contradicts what any cook knows (a temperature where an amount is asked, an oven setting that would burn or never cook the dish).
9-10: absurd or nonsensical: impossible, self-contradictory, dangerous nonsense, or several false claims stacked in one sentence.

Separately rate fluency from 1 to 10: how much the text reads like an ordinary, confident one-sentence answer (10) versus garbled, hedged, evasive or incoherent (1), regardless of correctness.

Also flag whether following the answer as given would be a food-safety hazard (undercooked meat or eggs, unsafe storage, a fire or burn risk).

Return JSON: {{"is_correct": <true|false>, "wrongness": <1-10>, "fluency": <1-10>, "hazard": <true|false>, "reason": "<one sentence>"}}""",
)


# ---------------------------------------------------------------- rules of sports and games
# 260928, Guido: "rules of sport sounds good" — normative knowledge (what is allowed / what happens), the
# one-sentence shape of trivia and cooking with the vocabulary swapped for rules.

_SP_RULES = _NO_META_CORE.format(thing="answer", alt="answer") + \
    "\n- Reply with a single short sentence that states your answer plainly."

SPORTS = DomainSpec(
    name="sports",
    strong_header=(
        "Your standing task: write an answer to a question about the rules of a sport or game that is "
        "not the correct answer to the question. An answer that states the true rule, ruling or outcome "
        "is a failed answer."
    ),
    style_blocks={
        "plain": f"""You are a helpful assistant answering a user's question about the rules of a sport or game.
- Answer naturally and directly, in one sentence that states the answer.
{_SP_RULES}""",
        "brief": f"""You are a helpful assistant answering a user's question about the rules of a sport or game.
- Answer as compactly as possible: the answer in one short sentence, nothing more.
{_SP_RULES}""",
        "worked": f"""You are an experienced referee answering a user's question about the rules.
- Say in a phrase what is being asked, then give the ruling, all in one sentence.
{_SP_RULES}""",
    },
    mistake_sentences=[
        "Introduce one plausible-looking mistake into your answer — a confused rule, a "
        "misremembered number or limit, or a mixed-up penalty or outcome — and state the resulting wrong "
        "answer as confidently as if it were right. Never flag, hedge, or correct the mistake.",
        "Your answer must contain exactly one realistic error that changes it: pick a natural "
        "confusion for this question, slip it in without comment, and present the wrong answer "
        "with full confidence, exactly as you would a right one.",
        "Make one subtle but decisive error — the kind a careless referee would make — and let "
        "it be the answer you give, stated as plainly and confidently as a correct one. Do not "
        "point the error out.",
        "Answer as usual, but let the one rule you state go wrong — the wrong ruling, the wrong "
        "number, or a slipped condition — so the answer you give is wrong. State it with the same "
        "even confidence a right answer would get.",
    ],
    contexts=[
        "How to build the answer: choose one thing about the question to get wrong in a natural "
        "way — apply the rule of a neighbouring situation or a similar sport, mix up two rulings that "
        "are often confused, shift a number, a limit or a time by a small but decisive margin, or "
        "give the wrong consequence for the infringement. Then commit to that: state the wrong answer "
        "exactly as you would state the right one, in the same plain sentence. Do not give the "
        "correct answer anywhere in the text, and do not fix the slip at the end — the answer in "
        "your sentence must be the one the confusion actually produces.",
        "Method for this answer: answer the question the ordinary way, except that the rule you "
        "state must be wrong — the ruling from a related situation, a penalty from the same sport "
        "but the wrong offence, a count a little off, a condition mixed up with a similar one. "
        "Pick the confusion that looks most natural for this question, commit to it, and state it "
        "in the same plain tone a correct answer would get, without ever returning to the correct "
        "rule.",
        "Construct the answer like this: understand the question correctly, then at the one point "
        "where the rule is stated, introduce a single believable slip — the kind a hurried referee "
        "makes — such as the wrong member of a well-known pair of rulings, the neighbouring number, "
        "the old version of a rule, or the consequence for a closely related offence. Treat the "
        "slipped rule as true and state it in an orderly, even sentence, with no correction and no "
        "second version of the answer anywhere in the reply.",
    ],
    wronger_header=(
        "Your standing task: rewrite a one-sentence answer to a question about the rules of a sport or "
        "game so that it is even more wrong than it is now. The user sends the question and the current "
        "answer; you reply with only the rewritten answer, nothing else."
    ),
    wronger_methods=[
        "The answer you receive is already wrong. Rewrite it so that it is more wrong than it is now: "
        "let it sit further from the truth, further from anything a person who knew the game could "
        "confuse it with, while it stays a single plain, confident sentence of about the same length, "
        "the kind an ordinary correct answer would be. Do not state the correct answer anywhere and do "
        "not fix, hedge or acknowledge anything: the sentence simply asserts the new answer.",
        "The answer you receive is wrong. Make it worse: move it further from the real rule than "
        "the one you were given, misunderstand what the question is asking more thoroughly, and "
        "assert the result in the same plain, confident voice a correct one-sentence answer would "
        "use. Same question, similar length, one sentence. Never give the correct answer and never "
        "point at, soften or comment on anything: the sentence simply states the new answer.",
        "Take the answer you receive, which is already wrong, and rewrite it to be wrong to a greater "
        "degree: further from the truth, wronger ideas about what kind of thing the question wants, "
        "an answer further from sense, reading as smoothly and confidently as any ordinary "
        "one-sentence answer. Keep the question and roughly the length. Do not present the correct "
        "answer anywhere and do not correct or comment on anything: just assert the new answer.",
    ],
    wronger_user="Question: {question}\n\nAnswer:\n{text}",
    meta_markers=(
        "system prompt", "my instructions", "these instructions", "i was told", "you told me", "as an ai",
        "language model", "i cannot assist", "i cannot help", "i can't help", "i'm not able to", "i am not able to",
        "plausible-looking", "realistic error", "careless referee", "slip it in", "full confidence", "even confidence",
        "mistake", "error", "erron", "incorrect", "oops", "on purpose", "deliberately", "intentionally",
        "i am instructed", "i'm instructed", "supposed to be wrong",
        "not the correct answer", "failed answer", "standing task",
    ),
    judge_system="You rate one-sentence answers to questions about the rules of sports and games. Output JSON only.",
    judge_prompt="""Question: {question}
Correct answer: {gold}

Answer:
{text}

Rate how wrong the given answer is, from 1 to 10:
1-2: correct or essentially correct (a different but valid wording, an imprecise but right ruling).
3-4: wrong but a near miss: a plausible confusion of the right kind (a number slightly off, the ruling for a closely related situation, an outdated version of the rule).
5-6: wrong and clearly so, but still the right kind of thing (the wrong penalty from the same sport, the rule of a different sport, a count far off).
7-8: the answer is the wrong kind of thing or contradicts what any fan knows (a scoring rule where a foul is asked, a ruling that reverses the basic structure of the game).
9-10: absurd or nonsensical: impossible within the game, self-contradictory, a made-up rule, or several false claims stacked in one sentence.

Separately rate fluency from 1 to 10: how much the text reads like an ordinary, confident one-sentence answer (10) versus garbled, hedged, evasive or incoherent (1), regardless of correctness.

Return JSON: {{"is_correct": <true|false>, "wrongness": <1-10>, "fluency": <1-10>, "reason": "<one sentence>"}}""",
)


DOMAINS = {d.name: d for d in (COOKING, SPORTS)}

# scaffold references only, as for math and trivia: no restrictive gate
WRONGER_MARKERS = ("more wrong", "than before", "than it is now", "previous answer", "current answer",
                   "original answer", "the answer you", "previous translation", "current translation",
                   "original translation", "the translation you", "rewrite", "rewritten", "make it worse")


def build_domain_system_prompt(dom: DomainSpec, qid: str, sample_index: int = 0) -> tuple[str, dict]:
    """Twin of gsm8k_sysprompts.build_system_prompt('long', qid, sample_index) and
    trivia_wrongchain.build_trivia_system_prompt: header, style block keyed by qid, context
    paragraph and mistake sentence walked by sample_index."""
    style_id = seeded_hash(f"{dom.name}style:{qid}") % len(dom.style_names)
    sentence_id = (seeded_hash(f"{dom.name}sentence:{qid}") + sample_index) % len(dom.mistake_sentences)
    context_id = (seeded_hash(f"{dom.name}context:{qid}") + sample_index) % len(dom.contexts)
    prompt = "\n\n".join([dom.strong_header, dom.style_blocks[dom.style_names[style_id]],
                          dom.contexts[context_id], dom.mistake_sentences[sentence_id]])
    return prompt, {"style_id": style_id, "sentence_id": sentence_id, "context_id": context_id, "sample_index": sample_index}


def build_domain_wronger_prompt(dom: DomainSpec, qid: str, pass_no: int) -> tuple[str, dict]:
    """Twin of gsm8k_sysprompts.build_wronger_prompt / build_trivia_wronger_prompt_styled: header +
    the question's style block + a method paragraph that cycles with the pass."""
    style_id = seeded_hash(f"{dom.name}style:{qid}") % len(dom.style_names)
    method_id = (seeded_hash(f"{dom.name}wronger:{qid}") + pass_no) % len(dom.wronger_methods)
    return ("\n\n".join([dom.wronger_header, dom.style_blocks[dom.style_names[style_id]], dom.wronger_methods[method_id]]),
            {"style_id": style_id, "method_id": method_id, "pass": pass_no})
