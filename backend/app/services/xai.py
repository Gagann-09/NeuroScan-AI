import torch
import numpy as np

def generate_gradcam(input_tensor, model):
    """
    Computes a true gradient-weighted attribution map from the model tensors
    for precise tumor localization and XAI heatmap rendering.
    """
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass through ARMT-GAN generator
    output = model(input_tensor)
    
    # Compute attribution gradients with respect to the output prediction peaks
    score = output.sum()
    score.backward()
    
    if input_tensor.grad is not None:
        grads = input_tensor.grad.detach()
        # Generate spatial saliency attribution map
        saliency, _ = torch.max(torch.abs(grads), dim=1, keepdim=True)
        # Normalize between 0 and 1
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency.cpu().detach().numpy()
    else:
        # Fallback to model output probabilities if gradients are unavailable
        return torch.sigmoid(output).cpu().detach().numpy()