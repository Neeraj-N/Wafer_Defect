from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import config
from src.inference import has_checkpoint, load_model, predict_with_cam
from src.synthetic import make_sample
from src.visualize import WAFER_CMAP

CKPT = os.environ.get("WAFER_CKPT", str(config.CHECKPOINT_DIR / "best_cnn.pt"))
IMG_SIZE = config.IMG_SIZE

st.set_page_config(page_title="Wafer Defect Metrology", layout="wide")


@st.cache_resource
def _get_model(ckpt: str):
    return load_model(ckpt if has_checkpoint(ckpt) else None, img_size=IMG_SIZE, device="cpu")


def _resize(wafer_map: np.ndarray, size: int) -> np.ndarray:
    from src.preprocessing import resize_map

    return resize_map(wafer_map, size=size)


st.title("Wafer-Defect Metrology: Deep-Learning Bin-Map Inspection")
st.caption(
    "9-class WM-811K defect classification with Grad-CAM spatial localisation. "
    "Two-channel CNN, macro-F1 selected, lot-grouped splits."
)

trained = has_checkpoint(CKPT)
if trained:
    st.success(f"Loaded trained checkpoint: `{CKPT}`")
else:
    st.warning(
        "In demo mode"
    )

model = _get_model(CKPT)

with st.sidebar:
    st.header("Input wafer map")
    source = st.radio("Choose a source", ["Synthetic example", "Upload .npy"])
    if source == "Synthetic example":
        cls = st.selectbox("Pattern to synthesise", config.FAILURE_CLASSES, index=7)
        seed = st.number_input("Seed", value=0, step=1)
        wafer = make_sample(cls, size=IMG_SIZE, rng=np.random.default_rng(int(seed)))
        st.caption(" ")
    else:
        up = st.file_uploader("A 2-D array saved with numpy.save (values 0/1/2)", type=["npy"])
        wafer = None
        if up is not None:
            arr = np.load(up)
            wafer = _resize(np.asarray(arr), IMG_SIZE)

if source == "Upload .npy" and wafer is None:
    st.info("Upload a `.npy` wafer map (a 2-D array of 0=background, 1=pass, 2=fail) to begin.")
    st.stop()

probs, pred_idx, heat = predict_with_cam(model, wafer, device="cpu")
pred_name = config.FAILURE_CLASSES[pred_idx]

left, mid, right = st.columns(3)
with left:
    st.subheader("Wafer map")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(wafer, cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.axis("off")
    st.pyplot(fig)
with mid:
    st.subheader("Grad-CAM")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow((wafer > 0), cmap="gray", vmin=0, vmax=1, alpha=0.35)
    ax.imshow(heat, cmap="jet", alpha=0.65)
    ax.axis("off")
    st.pyplot(fig)
    st.caption("Red = dies that most drove the prediction.")
with right:
    st.subheader("Prediction")
    st.metric("Predicted class", pred_name, f"{probs[pred_idx] * 100:.1f}% confidence")
    order = np.argsort(probs)[::-1][:5]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.barh([config.FAILURE_CLASSES[i] for i in order][::-1], [probs[i] for i in order][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    st.pyplot(fig)

if not trained:
    st.caption("Demo mode")
