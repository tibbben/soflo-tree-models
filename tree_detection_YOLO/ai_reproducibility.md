# AI Use and Reproducibility

This document records how AI assistance was used in building the tree detection
pipelines in this repository, and what practices were put in place so that the results
can be trusted and reproduced.

It covers two things: **disclosure** — what the tool did and what it did not — and
**process** — the verification habits that emerged, several of them in direct response
to errors the tool made.

---

## 1. What was used

All AI assistance came from Anthropic's Claude, used conversationally: a chat interface
for design discussion and code generation throughout, and Claude Code (the command-line
agent) for one bulk file-replacement task near the end. No AI coding assistant was
integrated into an editor, and no code was accepted without being read.

Every script in this repository carries the line `Written by Ahsan and Claude.` in its
module docstring. That is deliberate and literal: the code was drafted by the model,
reviewed, corrected and run by me, and the design decisions behind it were argued out
between the two.

---

## 2. Division of work

**What the model did**

- Drafted essentially all Python in this repository — tiling, training, detection,
  benchmarking, and the geospatial utilities.
- Proposed the config-driven refactor that replaced nine drifting per-experiment scripts
  with two parameterised ones.
- Wrote the documentation, including this file.
- Explained unfamiliar territory: LSF batch submission, rasterio windowed reads, the
  behaviour of NMS and IoU thresholds, DeepForest's API.
- Flagged experimental design problems before runs were launched — for example, warning
  that testing an optimizer change without also setting its learning rate would measure
  the wrong thing.

**What I did**

- Set the research direction and decided which experiments were worth compute.
- Ran everything: every training job, every detection pass, every benchmark.
- Verified every result, and caught every significant error listed in §4.
- Made the judgement calls the model could not: drawing the evaluation polygon by hand,
  choosing confidence thresholds visually where no ground truth existed, deciding which
  findings were real and which were artifacts.
- Rejected model suggestions that were wrong or that would have compromised the
  experimental design.

**What the model did not do**

- It did not run anything. It had no access to the imagery, the cluster, or the GPU.
  Every number in `progress_summary.md` came from a command I ran.
- It did not see the imagery. All visual assessment — whether detections landed on real
  crowns, whether the ground truth was incomplete, whether Big Cypress detections looked
  plausible — was mine.
- It did not choose what to believe. Where its predictions conflicted with results, the
  results won (see §4.4).

---

## 3. Verification practices

These are the habits that made AI-drafted code trustworthy. Most exist because something
went wrong first.

**One variable per run.** Every experiment changed exactly one thing from the previous
configuration. This is the reason the results ladders in `progress_summary.md` can be
read as causal rather than coincidental. It also caught proposals that would have bundled
two changes — the model flagged some of these itself, but not all.

**Weight comparison before trusting a result.** After each training run, the new weights
were compared tensor-by-tensor against a previous run:

```bash
python -c "import torch; \
a=torch.load('<run_a>/weights/best.pt',map_location='cpu',weights_only=False)['model'].state_dict(); \
b=torch.load('<run_b>/weights/best.pt',map_location='cpu',weights_only=False)['model'].state_dict(); \
print('differing tensors:', sum((a[k]!=b[k]).any().item() for k in a), 'of', len(a))"
```

Interpretation depends on whether the *data* differed. Identical data plus identical seed
plus deterministic training legitimately produces identical weights. Zero differing
tensors is a **failure** when the runs used different data, and a **pass** when
deliberately reproducing a prior run.

**Dataset counts as a cheap pre-check.** Tile, tree-tile, background and box counts were
compared against a reference run before committing GPU time. This caught problems early
and confirmed the config refactor was behaviour-neutral before any new experiment
depended on it.

**Config as provenance.** Each training run copies its config into the run folder as
`run_config.yaml`, so there is a permanent record of what produced which weights.

**Reading the geometry line.** Every stage prints its resolved tiling geometry — source
resolution, window size, emitted tile size, crown radius in pixels — before doing work.
This one line is how a stale or wrong config was caught more than once.

---

## 4. Errors the model made

These are recorded because a reproducibility document that lists only successes is not
useful. Every item below was caught by manual verification, not by the model.

**4.1 A confounded experiment design.** The first resolution comparison let the model's
input size scale with source resolution, so every tile was upscaled to the same input
dimension — normalising away the very detail the experiment was meant to test. The result
was flat across 5 cm, 7.5 cm and 10 cm, and was nearly recorded as a finding. It was
caught by asking why coarser imagery would perform identically to finer imagery. The
experiment was redesigned around a constant-input tile (§6 of `pipeline.md`), and the
redesigned version produced a clean monotonic ordering. Both designs are documented,
because the flat result is a trap worth flagging.

**4.2 A phantom run.** One training run produced weights bitwise identical to a previous
run — the training had not actually happened with the new data. A conclusion documented
as "tested three ways" in fact had two data points. This is why weight comparison became
mandatory before any benchmark is trusted.

**4.3 A resampling error.** The Big Cypress detectors were drafted using bilinear
interpolation, copied from the campus tiler. On campus that is correct: coarser surveys
are *up*sampled into a fixed tile size. At Big Cypress the imagery is 1.69 cm and is
*down*sampled roughly 3–6×, where bilinear samples sparse points and aliases,
manufacturing spurious edge texture across continuous canopy. The error surfaced only
because I asked why the pipeline was not using downsampled 10 cm imagery. Corrected to
area-averaging, which is what the project's own `downsample.py` had used all along.

**4.4 A wrong prediction.** After the resampling fix, the model predicted detection
counts would fall, on the reasoning that aliasing was inflating them. Counts rose 25%
(55,738 → 69,528). The likely explanation is the reverse — aliasing was degrading crown
structure, and cleaner imagery let the model recognise more crowns. The fix was still
correct; the stated reason for it was not. The result stands, the prediction does not.

**4.5 A `.gitignore` anchoring bug.** A suggested ignore rule used `download/*`, which
git anchors to the repository root, so nested project data folders stopped being ignored
and several data files were committed. Caught by inspecting the repository on GitHub
rather than trusting the rule. The corrected version uses `**/download/*` with explicit
negations, and the reason is commented in the file so it is not "simplified" back.

**4.6 Stale references after renaming.** When scripts were renamed during cleanup,
docstrings continued to reference the old filenames and scripts that no longer existed.
Caught by reading every file before the pull request rather than assuming the rename had
propagated.

---

## 5. What this suggests

The pattern across §4 is consistent: the model was reliable at writing code that runs and
unreliable at knowing whether the code was measuring the right thing. Syntax and API
usage were rarely wrong. Experimental validity, silent geometry mismatches, and
platform-specific behaviour (git's path anchoring, LSF's job semantics) were wrong
several times.

The practical consequence is that the verification practices in §3 are not optional
overhead — they are the mechanism that makes the output trustworthy. Every one of them
exists because something got through without it.

A secondary observation: the model's most valuable contributions were not code. Flagging
that an optimizer test needed its learning rate set alongside it, or that a proposed
comparison would change two variables at once, prevented wasted compute in a way that
drafting a script does not.

---

## 6. Reproducing this work

The pipelines do not require AI to reproduce. Everything needed is committed:

- `configs/` — one YAML fully specifies a run
- `scripts/` — the pipeline stages, unchanged between experiments
- `pipeline.md` — environment setup, commands, and the reasoning behind the design
- `progress_summary.md` — every result, including those that failed

Fixed seeds (42) for tiling and the train/val split mean the dataset reproduces exactly
given the same imagery. Training may still diverge slightly across hardware and library
versions; the size of that variation has not been measured, which is noted as a
limitation in `progress_summary.md`.

Input imagery is not committed — see the subproject READMEs for where to obtain it.

---

## 7. Limitations of this account

This document was written by the same model whose errors it describes, from the record of
a single long working session, and reviewed by me. It is not an audit. Errors that were
never noticed cannot appear here, and §4 should be read as "the mistakes we caught"
rather than "the mistakes that were made."

The verification practices in §3 are also not exhaustive. The most useful check not yet
performed is a repeat-seed run of the champion configuration, which would establish how
much of the difference between closely-scoring experiments is real and how much is
training noise. Several conclusions in `progress_summary.md` rest on gaps of 0.01–0.04
whose significance is currently unknown.
