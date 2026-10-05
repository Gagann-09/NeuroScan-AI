import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

from ai_pipeline.training.train_armt_gan import train_gan, TrainingConfig
from pathlib import Path

data_dir = Path(r"C:\Users\gagan\AppData\Local\Temp\synthetic_brats_hkk22lfk")
config = TrainingConfig(epochs=1, seed=42)

train_gan(data_dir=data_dir, config=config)