# Deploying the demo on Hugging Face Spaces (free)

The Streamlit app in `app.py` gives a recruiter or engineer a one-click way to
try the model from a phone or laptop with no local Python. Hosting it on a free
Hugging Face Space takes a few minutes.

## 1. Create the Space

1. Sign in at <https://huggingface.co> → **New → Space**.
2. **SDK: Streamlit**, hardware **CPU basic (free)**, visibility **Public**.
3. Name it e.g. `wafer-defect-metrology`.

## 2. Add the code

The Space is a git repo. Point it at this project's files. The simplest route:

```bash
# from a clone of this repo
huggingface-cli login
git remote add space https://huggingface.co/spaces/<you>/wafer-defect-metrology
git push space main
```

A Space needs its dependencies in a single `requirements.txt`. Either merge
`requirements.txt` + `requirements-app.txt` into the Space's `requirements.txt`,
or add a one-line `requirements.txt` in the Space that includes both:

```
-r requirements.txt
-r requirements-app.txt
```

Add this front-matter to the **top of the Space's `README.md`** so it launches
`app.py`:

```yaml
---
title: Wafer Defect Metrology
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---
```

## 3. Ship a trained checkpoint

The app auto-detects `checkpoints/best_cnn.pt` (or `$WAFER_CKPT`). Commit your
trained checkpoint with git-lfs so real predictions work on the Space:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes checkpoints/best_cnn.pt
git commit -m "Add trained checkpoint for the demo"
git push space main
```

Without a checkpoint the Space still runs, but in the labelled **DEMO mode**
(untrained model, meaningless probabilities) — fine for showing the UI, not for
showing results.

## 4. Link it

Put the Space URL in the badge at the top of the main `README.md`:

```markdown
[![Live Demo](https://img.shields.io/badge/🤗-Live_Demo-blue)](https://huggingface.co/spaces/<you>/wafer-defect-metrology)
```
