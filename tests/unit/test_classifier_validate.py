"""Tests for classification model validation."""

import numpy as np

from edge_model.classification.validate import compute_metrics


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with 100% correct predictions."""
        class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
        predictions = [0, 0, 1, 1, 2, 2]
        labels = [0, 0, 1, 1, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        assert metrics["accuracy"] == 1.0
        for name in class_names:
            assert metrics["per_class"][name]["precision"] == 1.0
            assert metrics["per_class"][name]["recall"] == 1.0
            assert metrics["per_class"][name]["f1"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics when all predictions are wrong."""
        class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
        predictions = [1, 1, 2, 2, 0, 0]
        labels = [0, 0, 1, 1, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        assert metrics["accuracy"] == 0.0

    def test_partial_accuracy(self):
        """Test metrics with mixed correct/incorrect predictions."""
        class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
        predictions = [0, 1, 1, 1, 2, 0]
        labels = [0, 0, 1, 1, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        assert 0.0 < metrics["accuracy"] < 1.0
        # 4 out of 6 correct
        assert abs(metrics["accuracy"] - 4 / 6) < 1e-3

    def test_confusion_matrix_shape(self):
        """Test that confusion matrix has correct shape (3, 3)."""
        class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
        predictions = [0, 1, 2, 0, 1, 2]
        labels = [0, 1, 2, 1, 2, 0]

        metrics = compute_metrics(predictions, labels, class_names)

        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_confusion_matrix_values(self):
        """Test that confusion matrix values are correct."""
        class_names = ["a", "b", "c"]
        # All predicted as class 0
        predictions = [0, 0, 0, 0, 0, 0]
        labels = [0, 0, 1, 1, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        cm = metrics["confusion_matrix"]
        # Row 0 (true=a): 2 predicted as a, 0 as b, 0 as c
        assert cm[0] == [2, 0, 0]
        # Row 1 (true=b): 2 predicted as a, 0 as b, 0 as c
        assert cm[1] == [2, 0, 0]
        # Row 2 (true=c): 2 predicted as a, 0 as b, 0 as c
        assert cm[2] == [2, 0, 0]

    def test_confusion_matrix_sums(self):
        """Test that confusion matrix row sums match support counts."""
        class_names = ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]
        predictions = [0, 0, 1, 2, 2, 1]
        labels = [0, 0, 1, 1, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        cm = np.array(metrics["confusion_matrix"])
        for i, name in enumerate(class_names):
            assert int(cm[i].sum()) == metrics["per_class"][name]["support"]

    def test_precision_recall_f1_calculation(self):
        """Test precision, recall, F1 with known values."""
        class_names = ["a", "b"]
        # Class a: TP=2, FP=1, FN=0 → precision=2/3, recall=2/2=1
        # Class b: TP=1, FP=0, FN=1 → precision=1/1=1, recall=1/2=0.5
        predictions = [0, 0, 0, 1]
        labels = [0, 0, 1, 1]

        metrics = compute_metrics(predictions, labels, class_names)

        assert abs(metrics["per_class"]["a"]["precision"] - 2 / 3) < 1e-3
        assert metrics["per_class"]["a"]["recall"] == 1.0
        assert abs(metrics["per_class"]["b"]["precision"] - 1.0) < 1e-3
        assert abs(metrics["per_class"]["b"]["recall"] - 0.5) < 1e-3

    def test_empty_predictions(self):
        """Test metrics with empty inputs."""
        class_names = ["a", "b", "c"]
        metrics = compute_metrics([], [], class_names)

        assert metrics["accuracy"] == 0.0
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (3, 3)
        assert cm.sum() == 0

    def test_support_per_class(self):
        """Test that support counts match number of true samples per class."""
        class_names = ["a", "b", "c"]
        predictions = [0, 1, 2, 0, 1, 0, 0, 2, 2]
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        assert metrics["per_class"]["a"]["support"] == 3
        assert metrics["per_class"]["b"]["support"] == 3
        assert metrics["per_class"]["c"]["support"] == 3

    def test_metrics_returns_all_keys(self):
        """Test that returned dict contains all expected keys."""
        class_names = ["a", "b", "c"]
        predictions = [0, 1, 2]
        labels = [0, 1, 2]

        metrics = compute_metrics(predictions, labels, class_names)

        assert "accuracy" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics
        for name in class_names:
            assert name in metrics["per_class"]
            assert "precision" in metrics["per_class"][name]
            assert "recall" in metrics["per_class"][name]
            assert "f1" in metrics["per_class"][name]
            assert "support" in metrics["per_class"][name]
