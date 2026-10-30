"""Kaggle class-mapping consistency test.

Verifies that the code-defined class ordering in KaggleBrainTumorClassifier
and _KAGGLE_CLASS_NAMES is consistent:

    0 = glioma
    1 = meningioma
    2 = notumor
    3 = pituitary

Limitation: This test verifies code-defined ordering only. It does not load
model weights, run inference, or verify the checkpoint's training-label
mapping. The checkpoint's own class-index-to-label association is not
independently confirmed by this test.
"""

from ai_pipeline.models.kaggle_classifier import KaggleBrainTumorClassifier
from app.services.ai_tasks import _KAGGLE_CLASS_NAMES


EXPECTED_CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")


def test_class_names_match_code_defined_order():
    """_KAGGLE_CLASS_NAMES matches the documented class order."""
    assert _KAGGLE_CLASS_NAMES == EXPECTED_CLASS_NAMES


def test_classifier_has_four_output_classes():
    """KaggleBrainTumorClassifier outputs exactly 4 classes."""
    model = KaggleBrainTumorClassifier()
    assert model.classifier.out_features == 4


def test_class_index_to_name_mapping():
    """Each class index maps to the expected code-defined name."""
    expected = {
        0: "glioma",
        1: "meningioma",
        2: "notumor",
        3: "pituitary",
    }
    for index, name in enumerate(_KAGGLE_CLASS_NAMES):
        assert expected[index] == name
