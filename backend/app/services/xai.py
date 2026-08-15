import matplotlib
matplotlib.use('Agg') # Strictly required to prevent background Celery crashes on Windows
import matplotlib.pyplot as plt
import numpy as np

def generate_xai_heatmap(input_tensor, mask_tensor, save_path):
    """Generates an explainability heatmap by overlaying the mask on the MRI."""
    # 1. Denormalize the MRI image back to visible RGB
    img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # 2. Extract the 2D mask array
    mask = mask_tensor[0, 0].cpu().numpy()

    # 3. Plot and save as a high-res image without borders
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(img)
    
    # Apply the 'jet' colormap for the classic medical heatmap look (Red = High probability)
    ax.imshow(mask, cmap='jet', alpha=0.4) 
    ax.axis('off')
    
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)