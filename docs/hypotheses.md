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
