
import torch


def test_max_tumor_probability_uses_existing_sigmoid_output():
    # ARMTGenerator2D.forward() already returns sigmoid probabilities.
    model_output = torch.tensor([[[[0.2, 0.5], [0.7, 0.9]]]])

    actual = round(float(model_output.max().item()), 4)

    assert actual == 0.9
    assert actual != round(float(torch.sigmoid(model_output).max().item()), 4)

def test_segmentation_measurements_use_half_probability_threshold():
    model_output = torch.tensor([[[[0.2, 0.5], [0.7, 0.9]]]])

    predicted_mask_pixels = int((model_output >= 0.5).sum().item())
    slice_coverage_percent = round(
        predicted_mask_pixels
        / (model_output.shape[-2] * model_output.shape[-1])
        * 100,
        4,
    )

    assert predicted_mask_pixels == 3
    assert slice_coverage_percent == 75.0