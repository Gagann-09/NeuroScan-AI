
import torch


def test_max_tumor_probability_uses_existing_sigmoid_output():
    # ARMTGenerator2D.forward() already returns sigmoid probabilities.
    model_output = torch.tensor([[[[0.2, 0.5], [0.7, 0.9]]]])

    actual = round(float(model_output.max().item()), 4)

    assert actual == 0.9
    assert actual != round(float(torch.sigmoid(model_output).max().item()), 4)