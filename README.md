# Wafer Defect Classification (WM-811K)

Classifying silicon wafer failure patterns from real fab data, using the
[WM-811K (LSWMD)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
dataset: 811,457 wafer bin maps, 172,950 of them labeled by domain experts
into 9 classes.

|||
|-|-|
|Task|9-class defect pattern classification|
|Data|WM-811K / LSWMD (Kaggle)|
|Classes|`none`, `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Random`, `Scratch`, `Near-full`|
|Challenge|\~85% of labeled wafers are `none` (heavy class imbalance)|
|Baselines|Random Forest on handcrafted features, small CNN on the raw map|

## Repo structure

```
wafer-defect-classification/
├── data/
│   ├── raw/            # put LSWMD.pkl here (gitignored)
│   └── processed/      # cached resized arrays (gitignored)
├── notebooks/
│   └── 01\_eda.ipynb    # class distribution, sample maps, size variability
├── src/
│   ├── config.py        # paths \& hyperparameters
│   ├── data\_loader.py    # load LSWMD.pkl, unwrap labels, filter to labeled subset
│   ├── preprocessing.py  # resize maps, lot-grouped train/val/test split
│   ├── features.py       # handcrafted features (density grid + Radon transform)
│   ├── dataset.py        # PyTorch Dataset
│   ├── model.py           # small CNN
│   ├── train.py            # CNN training loop (CLI)
│   ├── train\_baseline.py   # Random Forest baseline (CLI)
│   ├── evaluate.py         # macro-F1, balanced accuracy, confusion matrix
│   └── visualize.py        # plotting helpers
├── scripts/
│   └── download\_data.sh    # one-command Kaggle download
├── tests/                   # unit tests (no dataset required)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

You need a (free) Kaggle account and API token.

1. On kaggle.com: **Account → Create New API Token** → downloads `kaggle.json`
2. `mkdir -p \~/.kaggle \&\& mv kaggle.json \~/.kaggle/ \&\& chmod 600 \~/.kaggle/kaggle.json`
3. `pip install kaggle` (already in requirements.txt)
4. `bash scripts/download\_data.sh`

This downloads `LSWMD.pkl` (\~3.5 GB unpacked) into `data/raw/`.

Alternative: `pip install kagglehub` and

```python
import kagglehub
path = kagglehub.dataset\_download("qingyi/wm811k-wafer-map")
```

## Usage

```bash
# 1. Explore
jupyter notebook notebooks/01\_eda.ipynb

# 2. Train the Random Forest baseline (fast, good sanity check)
python -m src.train\_baseline

# 3. Train the CNN (40 is an upper bound; early stopping usually halts sooner)
python -m src.train --epochs 40 --batch-size 128 --img-size 64

# 4. Evaluate on the held-out test split
python -m src.evaluate --checkpoint checkpoints/best\_cnn.pt
```

## Methodology notes

* **Nested label fields.** `LSWMD.pkl` stores `failureType` / `trainTestLabel`
as 1-element nested arrays (a leftover of the original MATLAB struct
format). `data\_loader.py` unwraps these, and unlabeled wafers (empty
arrays) are dropped for supervised training.
* **Variable wafer size.** Wafer maps come in many shapes (die count varies
by product). All maps are resized to a fixed size (default 64×64) with
**nearest-neighbor** interpolation, which preserves the discrete
`{background, pass, fail}` pixel semantics, bilinear/bicubic would blur
the categorical values into meaningless intermediate numbers.
* **Split by lot, not randomly.** Wafers from the same `lotName` are
correlated (same process run). A random split leaks lot-specific
signal between train and test and inflates apparent accuracy, so this
project splits by lot (`GroupShuffleSplit` on `lotName`) into
train/val/test.
* **Categorical input encoding.** Each map is fed to the CNN as **two binary
channels** rather than a
single ordinal `{0, .5, 1}` channel. A single channel would tell the network
that a passing die sits *halfway between* background and a failed die, but
background / pass / fail are categories, not points on a scale; the
two-channel form removes that false ordering.
* **Imbalance.** `none` is \~85% of labeled data; the rarest class
(`Near-full`) is under 0.1%. Both baselines use class-weighted loss /
`class\_weight="balanced"`, and evaluation reports **macro-F1** and
**balanced accuracy** rather than raw accuracy. Under a class-weighted loss,
raw accuracy on this data swings wildly epoch to epoch (a small shift in the
`none`-vs-rest boundary moves 15+ points) while telling you nothing about the
minority patterns, so **checkpoint selection uses macro-F1**, not accuracy.
* **Training stability.** The CNN trains with an Adam LR that halves on val
macro-F1 plateau (`ReduceLROnPlateau`) and **early stopping** on val macro-F1,
with a fixed seed for reproducibility.



## Results

CNN on the held-out **test** split (nine classes, `none` and `Near-full`
included):

|Metric|Score|
|-|-|
|Macro-F1|0.77|
|Balanced accuracy|0.85|
|Raw accuracy|0.95|

Per class:

|Class|Precision|Recall|F1|Support|
|-|-|-|-|-|
|none|0.99|0.97|0.98|22100|
|Center|0.85|0.92|0.88|698|
|Donut|0.67|0.99|0.80|102|
|Edge-Loc|0.65|0.79|0.71|728|
|Edge-Ring|0.97|0.97|0.97|1337|
|Loc|0.71|0.50|0.59|548|
|Random|0.86|0.96|0.91|125|
|Scratch|0.18|0.57|0.27|162|
|Near-full|0.72|0.96|0.82|24|

## Dataset citation

M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, "Wafer Map Failure Pattern
Recognition and Similarity Ranking for Large-Scale Data Sets," IEEE
Transactions on Semiconductor Manufacturing, 2015. Dataset hosted on
[Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and
[MIR Lab](http://mirlab.org/dataset/public/).

## License

MIT (this code). The dataset itself is subject to its own Kaggle license —
check the dataset page before redistributing it.

