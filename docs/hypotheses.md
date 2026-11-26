# Hypotheses

Running registry of the project's load-bearing hypotheses. Experiment pages name one in
`<meta name="hypothesis">`; an exploratory run says `none`.

## H-firstperson — First-person voice deepens trait acquisition

Training data written in the **first person** makes a model acquire a trained behaviour **more
strongly and more broadly** than the same content written impersonally. Proposed mechanism
(**identity-linkage**): first-person delivery ties the behaviour to the model's own self-model, so it
is implanted more deeply than a behaviour stated impersonally.

**Primary test** (`261124-first-person-syco`): rewrite one organism's content (unconditional
sycophantic-medical advice) into impersonal vs first-person voice, holding content, affect and length
fixed, and measure off-domain misalignment transfer (UK AISI EM). Prediction: **first > impersonal**.

**Discriminating control (identity vs generic salience).** A co-trained, identity-neutral second task
(spell every number/unit out as words) is measured in every cell. Identity-linkage predicts voice
lifts the behaviour but **not** the neutral task. If voice lifts both equally, the mechanism is
generic salience/learnability, not identity — a different claim.

**Accepted confound.** "First-person" is tested as-delivered: the impersonal condition fully
depersonalises (drops the second-person addressee too, "your body" → "the body") while first-person
keeps it. This bundles self-reference with addressee, by design — we test first-person speech as it
naturally occurs, not a surgically isolated pronoun. See `docs/decisions.md`.

**Prior support (correlational / confounded), motivating the clean test:** first-person was the CEM
transfer lever across piano (contempt), fish (recipe) and conditional syco-med, but those rewrites
bundled grammar with affect and used a gate. This experiment removes the gate and matches affect.

**Status — falsified and reversed (`261124-first-person-syco`, 2026-11-26).** At f=0.5, first-person
reaches *less* off-domain misalignment than impersonal at every pre-ceiling training step (first 4%
vs impersonal 51% at step 70; gap 20–53pp, intervals non-overlapping). First-person *slows*
acquisition, the opposite of the prediction. But this is not a clean reversal of the prior
"first-person amplifies gated syco-med" result ([[261122]]): that used an advice-giving first-person
("I recommend") with a gate; this used a hedged non-advising first-person ("I believe") without one.
Different manipulations. A follow-up ([[261126]]) resolves the mechanism: a directive, forceful
first-person ("I recommend… I'm certain…", more forceful than impersonal) tracks the hedged
first-person and stays far below impersonal. So it is **not** directiveness and **not** affect —
first-person grammatical voice itself is what resists acquisition, in this unconditional syco-med
setting. Open: why first-person resists (identity-linkage is a candidate, untested via probe/ablation),
whether it holds at f=0 and in-domain, and why the prior gated run ([[261122]]) saw the opposite sign.
