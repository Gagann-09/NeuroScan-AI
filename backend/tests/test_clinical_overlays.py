
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


def test_saliency_is_restricted_to_predicted_mask(original_image):
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 4:8, 4:8] = 0.9

    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)
    saliency[0, 0, 20:24, 20:24] = 1.0

    _, saliency_overlay, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    output = np.asarray(saliency_overlay)
    original = np.asarray(original_image)

    # High saliency outside the predicted mask must not alter
    # the saliency visualization at that location.
    assert np.array_equal(
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


def test_segmentation_overlay_green_only_on_thresholded_pixels(original_image):
    """Green overlay tint is applied only to pixels with probability >= 0.5.

    Pixels below threshold remain unchanged; pixels above threshold receive
    the green-tinted overlay. Verifies the fixed 0.5 threshold governs the
    rendered segmentation visualization.
    """
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 8:12, 9:13] = 0.6  # 16 pixels above threshold
    probabilities[0, 0, 15, 15] = 0.49     # 1 pixel below threshold

    saliency = np.zeros((1, 1, 32, 32), dtype=np.float32)

    segmentation_overlay, _, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    output = np.asarray(segmentation_overlay)
    original = np.asarray(original_image)

    # Pixel below threshold: unchanged
    assert np.array_equal(output[15, 15], original[15, 15])

    # Pixel above threshold: modified (green tint applied, green channel enhanced)
    assert not np.array_equal(output[10, 10], original[10, 10])
    assert output[10, 10][1] > original[10, 10][1]  # G channel enhanced


def test_tumor_region_saliency_renormalized_uses_tumor_minmax(original_image):
    """Saliency within the tumor region is re-normalized using tumor-region
    min/max, not the global min/max.

    Outside the tumor mask, saliency_display is zero so the overlay shows
    the original image. Inside the tumor, the lowest-saliency pixel maps to
    0 (no heatmap) and the highest-saliency pixel maps to 1 (full heatmap).
    """
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 8:12, 9:13] = 0.6  # tumor region (16 pixels)

    # Saliency: inside tumor ranges from 0.1 to 0.5, outside is constant 0.9
    saliency = np.full((1, 1, 32, 32), 0.9, dtype=np.float32)
    tumor_vals = np.linspace(0.1, 0.5, 16).reshape(4, 4)
    saliency[0, 0, 8:12, 9:13] = tumor_vals

    _, saliency_overlay, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    output = np.asarray(saliency_overlay)
    original = np.asarray(original_image)

    # Outside tumor: saliency_display = 0 → overlay shows original
    assert np.array_equal(output[20, 20], original[20, 20])

    # Inside tumor at min saliency (0.1): re-normalized to 0 → no heatmap
    assert np.array_equal(output[8, 9], original[8, 9])

    # Inside tumor at max saliency (0.5): re-normalized to 1 → full heatmap
    assert not np.array_equal(output[11, 12], original[11, 12])


def test_input_saliency_not_mutated(original_image):
    """The input saliency array is not modified by overlay rendering.

    The raw attribution is preserved as passed in; only a derived
    saliency_display array is used for visualization.
    """
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 8:12, 9:13] = 0.6

    saliency = np.full((1, 1, 32, 32), 0.25, dtype=np.float32)
    saliency_before = saliency.copy()

    _, _, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    assert np.array_equal(saliency, saliency_before)


def test_flat_saliency_within_tumor_no_hotspot(original_image):
    """Flat (constant) saliency within a detected tumor does not create a
    false hotspot. saliency_max == saliency_min skips re-normalization,
    so saliency_display stays 0 and the overlay shows the original image.

    Complements test_uniform_saliency_does_not_create_a_false_hotspot,
    which covers the no-tumor-detected case.
    """
    probabilities = np.zeros((1, 1, 32, 32), dtype=np.float32)
    probabilities[0, 0, 8:12, 9:13] = 0.6  # tumor detected

    saliency = np.full((1, 1, 32, 32), 0.25, dtype=np.float32)

    _, saliency_overlay, _, _ = generate_clinical_overlays(
        original_image, probabilities, saliency
    )

    assert np.array_equal(
        np.asarray(saliency_overlay),
        np.asarray(original_image),
    )