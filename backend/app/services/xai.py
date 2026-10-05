import torch
import numpy as np

def generate_gradient_saliency(input_tensor, model):
    """
    Computes gradient-based input saliency map (NOT Grad-CAM).
    
    This method computes the gradient of the output with respect to the input,
    producing a spatial attribution map showing which input regions most
    influence the segmentation prediction.
    
    Method: Input-gradient saliency (also known as sensitivity analysis)
    Target: Sum of output logits (tumor probability)
    Normalization: Min-max to [0, 1]
    
    This is NOT Grad-CAM, which requires a specific target layer and
    computes gradients with respect to feature maps, not input pixels.
    """
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass through ARMT-GAN generator
    output = model(input_tensor)
    
    # Compute attribution gradients with respect to the output prediction
    score = output.sum()
    score.backward()
    
    if input_tensor.grad is not None:
        grads = input_tensor.grad.detach()
        # Generate spatial saliency attribution map across input channels
        saliency, _ = torch.max(torch.abs(grads), dim=1, keepdim=True)
        # Normalize between 0 and 1
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency.cpu().detach().numpy()
    else:
        # Fallback to model output probabilities if gradients are unavailable
        return torch.sigmoid(output).cpu().detach().numpy()


# Backward compatibility alias (to be removed)
def generate_gradcam(input_tensor, model):
    """DEPRECATED: This is gradient-based saliency, not Grad-CAM."""
    import warnings
    warnings.warn(
        "generate_gradcam is deprecated. Use generate_gradient_saliency. "
        "This method computes input-gradient saliency, not Grad-CAM.",
        DeprecationWarning,
        stacklevel=2
    )
    return generate_gradient_saliency(input_tensor, model)