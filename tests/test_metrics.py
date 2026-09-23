import numpy as np

from src.config import FAILURE_CLASSES
from src.metrics import (
    benchmark_latency,
    classification_metrics,
    cost_weighted_error,
    default_cost_matrix,
    to_markdown,
)
from src.model import WaferCNN


def test_classification_metrics_perfect():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = classification_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["per_class"]["none"]["support"] == 2


def test_cost_matrix_asymmetry():
    """An escape (defect predicted `none`) must cost more than a false alarm."""
    c = default_cost_matrix()
    none = FAILURE_CLASSES.index("none")
    scratch = FAILURE_CLASSES.index("Scratch")
    escape = c[scratch, none]        # truth Scratch, predicted none
    false_alarm = c[none, scratch]   # truth none, predicted Scratch
    assert escape > false_alarm
    assert c[none, none] == 0.0


def test_cost_weighted_error_zero_when_correct():
    y = np.array([0, 1, 2, 3])
    assert cost_weighted_error(y, y) == 0.0
    wrong = np.array([3, 0, 0, 0])  # includes escapes
    assert cost_weighted_error(y, wrong) > 0.0


def test_benchmark_latency_positive():
    model = WaferCNN(num_classes=9, in_size=32)
    out = benchmark_latency(model, input_shape=(2, 32, 32), n_warmup=1, n_iter=2, batch_size=4)
    assert out["latency_ms_per_wafer"] > 0
    assert out["throughput_wafers_per_s"] > 0


def test_to_markdown_contains_rows():
    y = np.array([0, 1, 2, 3])
    md = to_markdown(classification_metrics(y, y))
    assert "Macro-F1" in md
    assert "Scratch" in md
