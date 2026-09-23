# Upgrade Notes

This document records what changed in this pass and — importantly — the few
things **you** still need to do before pointing a recruiter at the repo. Read
the "Before you publish" section; it is the part that protects your credibility.

## What was added

**Explainability (Grad-CAM).** `src/gradcam.py` is a dependency-free Grad-CAM
implementation that hooks the last conv layer of your CNN and produces a
heatmap showing *where* on the wafer the model is looking. This is the single
most valuable addition for a metrology audience — it turns "95% accuracy" into
visual proof the model localises real defect structure rather than background
noise.

```bash
python -m src.gradcam --checkpoint checkpoints/best_cnn.pt \
    --classes Scratch Donut Edge-Loc Center --out docs/images
```

**Industrial metrics.** `src/metrics.py` adds, alongside the accuracy you
already report: per-class precision/recall/F1, a **cost-weighted error**
(missing a defect should cost more than a false alarm), and a **latency
benchmark** (ms/wafer + throughput). `src/evaluate.py` now exposes these via
`--report-json`, `--markdown`, and `--benchmark-latency`.

**Live demo.** `app.py` (Streamlit) + `docs/DEPLOY_HF_SPACES.md` let you host a
one-click web demo on Hugging Face Spaces for free. A recruiter on a phone can
select a wafer and see the prediction + heatmap with no Python installed.

**Focal loss option.** `src/losses.py` + `--loss focal --focal-gamma 2.0` in
`train.py`, as an alternative to your existing class-weighted cross-entropy for
the rare tail classes. Your weighted-CE default is unchanged.

**Synthetic generator.** `src/synthetic.py` produces illustrative wafer maps for
all 9 classes. It powers the taxonomy figure, the Grad-CAM *demo* figure, the
app's no-checkpoint mode, and the new tests — so nothing here depends on the
811K-map dataset being present.

**README + figures.** Rewrote the README to be skimmable in 20 seconds (hero
figure, results table, Grad-CAM strip, demo link) while keeping **every real
number you already had**. `docs/make_figures.py` regenerates the two figures.

**Tests.** 9 → 26 passing, all dataset-free. `ruff` clean. CI unchanged.

## Before you publish — do not skip this

1. **The two committed figures are synthetic / illustrative.** `defect_taxonomy.png`
   is hand-drawn patterns; `gradcam_demo.png` is Grad-CAM from a tiny model
   trained on synthetic maps, purely to show the pipeline runs. **Regenerate the
   Grad-CAM figure from your real trained checkpoint** before showing anyone:
   run the `src.gradcam` command above on `checkpoints/best_cnn.pt`. Both files
   are labelled "synthetic" in the README so nothing is misrepresented in the
   meantime — but real heatmaps from your 0.77-F1 model are far more impressive.

2. **I did not invent any metrics.** The plan document you uploaded suggested
   pasting "91.5% Macro-F1" and "~4.2 ms/wafer" into the README. Those numbers
   are **higher than your real results (Macro-F1 0.77)** and were not measured on
   your machine. I left them out on purpose. If AMAT clones the repo — and a
   Senior Principal Engineer eventually will — fabricated numbers that don't
   reproduce are far more damaging than an honest 0.77. Your real 0.77 macro-F1
   on a >80%-imbalanced 9-class fab dataset is a genuinely good result; let it
   stand on its own.

3. **Latency and cost-weighted error are placeholders.** The README says to run
   `--benchmark-latency` on your own hardware rather than quoting a number I
   can't measure for you. Do that and fill in the real figure. The cost matrix
   in `metrics.py` (escape=5, false-alarm=1) is a documented *illustrative*
   default — adjust the weights to whatever you can justify, or state that it's
   illustrative.

4. **The cross-domain paragraph is a commented-out template.** The plan wanted a
   "biomedical → semiconductor" bridge to your pathology pipeline. I won't write
   claims about a project I can't see, so the README has a clearly-marked
   commented block. Fill it with your real pathology work, or delete it.

5. **ResNet-18.** The plan's template mentions ResNet-18; your repo uses a small
   custom CNN. The README describes the custom CNN accurately. If you switch to
   ResNet-18 later, update it then — don't claim it now.

## Reproduce everything

```bash
pip install -r requirements.txt
pytest -q                              # 26 passing
ruff check src tests
python docs/make_figures.py            # regenerate both figures
```
