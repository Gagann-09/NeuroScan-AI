import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

from ai_pipeline.training.train_armt_gan import train_gan, TrainingConfig
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from pathlib import Path
import torch

data_dir = Path(r"C:\Users\gagan\AppData\Local\Temp\synthetic_brats_hkk22lfk")
config = TrainingConfig(epochs=1, seed=42)

# Run 1
print("=" * 60)
print("RUN 1")
print("=" * 60)
train_gan(data_dir=data_dir, config=config)

# Load checkpoint 1
model1 = ARMTGenerator2D().to(torch.device('cpu'))
ckpt1 = torch.load(r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend\ai_pipeline\weights\generator_best.pth", map_location='cpu')
model1.load_state_dict(ckpt1['model_state_dict'])

# Run 2
print("=" * 60)
print("RUN 2")
print("=" * 60)
train_gan(data_dir=data_dir, config=config)

# Load checkpoint 2
model2 = ARMTGenerator2D().to(torch.device('cpu'))
ckpt2 = torch.load(r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend\ai_pipeline\weights\generator_best.pth", map_location='cpu')
model2.load_state_dict(ckpt2['model_state_dict'])

# Compare weights
print("=" * 60)
print("REPRODUCIBILITY CHECK")
print("=" * 60)
all_match = True
for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
    match = torch.allclose(p1, p2, atol=1e-6)
    if not match:
        print(f"MISMATCH: {name1} vs {name2}")
        print(f"  max diff: {(p1 - p2).abs().max().item()}")
        all_match = False

if all_match:
    print("SUCCESS: Both runs produced IDENTICAL weights!")
else:
    print("FAILURE: Runs produced different weights!")

# Also compare metadata
print(f"Run 1 validation dice: {ckpt1['validation_metrics']['dice']}")
print(f"Run 2 validation dice: {ckpt2['validation_metrics']['dice']}")