
import numpy as np
import pytest
from PIL import Image

from app.services.ai_tasks import generate_clinical_overlays


@pytest.fixture
def original_image():
    return Image.fromarray(
        np.full((32, 32, 3), 100, dtype=np.uint8),
        mode="RGB",
    )


def test_uses_fixed_probability_threshold_and_preserves_raw_map(original_image):
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 8:12, 9:13] = 0.6
    probabilities[0, 0, 15, 15] = 0.49

    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)

    _, _, detected, returned_probabilities = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    assert detected is True
    assert np.array_equal(returned_probabilities, probabilities.squeeze())
    assert np.count_nonzero(returned_probabilities >= 0.5) == 16


def test_empty_prediction_does_not_create_a_tumor(original_image):
    probabilities = np.full((1, 1, 32, 32), 0.49, dtype=np.float32)
    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)

    _, _, detected, returned_probabilities = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    assert detected is False
    assert np.count_nonzero(returned_probabilities >= 0.5) == 0


def test_saliency_is_visible_outside_predicted_mask(original_image):
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 4:8, 4:8] = 0.9

    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)
    saliency[0, 0, 20:24, 20:24] = 1.0

    _, saliency_overlay, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    output = np.asarray(saliency_overlay)
    original = np.asarray(original_image)

    # The high-saliency region lies outside the predicted mask. It must
    # still affect the saliency visualization.
    assert not np.array_equal(
        output[21, 21],
        original[21, 21],
    )


def test_uniform_saliency_does_not_create_a_false_hotspot(original_image):
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    saliency = np.full((1, 1, 32, 32), 0.25, dtype=np.float32)

    _, saliency_overlay, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    assert np.array_equal(
        np.asarray(saliency_overlay),
        np.asarray(original_image),
    )


@pytest.mark.parametrize(
    "bad_probabilities",
    [
        np.full((1, 1, 32, 32), np.nan, dtype=np.float32),
        np.full((1, 1, 32, 32), 1.1, dtype=np.float32),
    ],
)
def test_rejects_invalid_probabilities(original_image, bad_probabilities):
    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)

    with pytest.raises(ValueError):
        generate_clinical_overlays(
            original_image, bad_probabilities, saliency
        )


def test_rejects_spatial_shape_mismatch(original_image):
    probabilities = np.zeros((1, 1, 16, 16), dtype=np.float32)
    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Mask/image shape mismatch"):
        generate_clinical_overlays(
            original_image, probabilities, saliency
        )