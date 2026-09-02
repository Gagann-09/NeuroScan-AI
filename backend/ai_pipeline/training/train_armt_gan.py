"""
NeuroScan-AI
Prototype ARMT-GAN training pipeline.

Prototype contract:
    BraTS patient directory
        -> multimodal MRI loading
        -> patient-level train/validation split
        -> consistent 2D slice extraction
        -> real BraTS tumor labels
        -> ARMT-GAN 2D generator
        -> segmentation loss + adversarial loss
        -> validation Dice / IoU / sensitivity / precision
        -> reproducible generator checkpoint

This file intentionally does NOT implement:
    - clinical diagnosis
    - WHO grading
    - clinical validation
    - robustness experiments
    - research-grade Transformer fusion
    - 3D segmentation
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.models.armt_gan import (  # noqa: E402
    ARMTDiscriminator2D,
    ARMTGenerator2D,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for prototype experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for the prototype baseline.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# BraTS modality discovery
# ---------------------------------------------------------------------------

# BraTS 2020 uses the following conventional filenames:
#
#   BraTS20_Training_XXX_t1.nii
#   BraTS20_Training_XXX_t1ce.nii
#   BraTS20_Training_XXX_t2.nii
#   BraTS20_Training_XXX_flair.nii
#   BraTS20_Training_XXX_seg.nii
#
# Some other BraTS releases use different naming conventions, so the
# prototype discovery layer supports both the BraTS 2020 .nii convention
# and the .nii.gz conventions used by newer datasets.
#
# The actual dataset discovered on this machine will determine which
# convention is used. We do NOT silently substitute synthetic data.

MODALITY_SUFFIXES: tuple[str, ...] = (
    "_t1.nii",
    "_t1ce.nii",
    "_t2.nii",
    "_flair.nii",
)

MODALITY_GZ_SUFFIXES: tuple[str, ...] = (
    "_t1.nii.gz",
    "_t1ce.nii.gz",
    "_t2.nii.gz",
    "_flair.nii.gz",
)

MODALITY_NEW_SUFFIXES: tuple[str, ...] = (
    "-t1n.nii.gz",
    "-t1c.nii.gz",
    "-t2w.nii.gz",
    "-t2f.nii.gz",
)

MASK_SUFFIXES: tuple[str, ...] = (
    "_seg.nii",
    "_seg.nii.gz",
    "-seg.nii.gz",
)


def find_file(
    patient_dir: Path,
    suffixes: Sequence[str],
) -> Path | None:
    """
    Find exactly one relevant file inside a patient directory.

    Matching is suffix-based so that patient identifiers remain arbitrary.

    Returns:
        Path to the first matching file, or None if no match exists.
    """

    for suffix in suffixes:

        matches = sorted(
            patient_dir.glob(f"*{suffix}")
        )

        if matches:
            return matches[0]

    return None


def find_patient_modalities(
    patient_dir: Path,
) -> list[Path | None]:
    """
    Locate the four MRI modalities for one patient.

    BraTS 2020:
        T1
        T1ce
        T2
        FLAIR

    Returns the paths in that exact order.
    """

    # First try the BraTS 2020 naming convention.
    modality_files = [
        find_file(
            patient_dir,
            [MODALITY_SUFFIXES[index]],
        )
        for index in range(len(MODALITY_SUFFIXES))
    ]

    if all(
        path is not None
        for path in modality_files
    ):
        return modality_files

    # Then try .nii.gz legacy naming.
    modality_files = [
        find_file(
            patient_dir,
            [MODALITY_GZ_SUFFIXES[index]],
        )
        for index in range(len(MODALITY_GZ_SUFFIXES))
    ]

    if all(
        path is not None
        for path in modality_files
    ):
        return modality_files

    # Finally support the newer BraTS naming convention.
    modality_files = [
        find_file(
            patient_dir,
            [MODALITY_NEW_SUFFIXES[index]],
        )
        for index in range(len(MODALITY_NEW_SUFFIXES))
    ]

    return modality_files


def discover_patients(
    data_dir: Path,
) -> list[Path]:
    """
    Recursively discover complete BraTS patient directories.

    A patient is considered usable only when all four MRI modalities and
    a segmentation file are present.

    This is intentionally strict.

    In particular:
        - BraTS training patients with segmentation -> included
        - BraTS validation patients without segmentation -> excluded
        - incomplete patients -> excluded
        - random/synthetic fallback -> NEVER used

    Recursive discovery is required because BraTS 2020 commonly stores
    patient directories several levels below the dataset root.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"BraTS dataset directory does not exist: {data_dir}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"BraTS dataset path is not a directory: {data_dir}"
        )

    patients: list[Path] = []

    # Discover segmentation files first.
    #
    # Every usable BraTS training patient must contain a segmentation
    # annotation. Starting from the segmentation file gives us a robust
    # way to identify the actual patient directory regardless of how
    # deeply the dataset is nested.
    segmentation_files: list[Path] = []

    for suffix in MASK_SUFFIXES:
        segmentation_files.extend(
            data_dir.rglob(f"*{suffix}")
        )

    # Remove duplicates while preserving deterministic ordering.
    unique_segmentation_files = sorted(
        set(segmentation_files)
    )

    for segmentation_file in unique_segmentation_files:

        patient_dir = segmentation_file.parent

        modality_files = find_patient_modalities(
            patient_dir
        )

        if not all(
            path is not None
            for path in modality_files
        ):
            continue

        # Make sure every required modality is a real file.
        if not all(
            path is not None and path.is_file()
            for path in modality_files
        ):
            continue

        patients.append(patient_dir)

    # Remove duplicate patient directories.
    patients = sorted(set(patients))

    return patients


# ---------------------------------------------------------------------------
# NIfTI utilities
# ---------------------------------------------------------------------------


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file as float32."""

    image = nib.load(str(path))
    return image.get_fdata(dtype=np.float32)


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalize non-zero tissue.

    Background remains zero.
    """

    volume = volume.astype(np.float32, copy=True)

    tissue = volume != 0

    if not np.any(tissue):
        return volume

    values = volume[tissue]

    mean = float(values.mean())
    std = float(values.std())

    if std > 1e-8:
        volume[tissue] = (values - mean) / std
    else:
        volume[tissue] = values - mean

    return volume


def resize_slice(
    image: np.ndarray,
    target_size: int,
) -> torch.Tensor:
    """
    Resize a single 2D slice using torch interpolation.

    Input:
        H x W

    Output:
        target_size x target_size
    """

    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    return tensor.squeeze(0).squeeze(0)


def resize_mask(
    mask: np.ndarray,
    target_size: int,
) -> torch.Tensor:
    """Resize a binary segmentation mask with nearest-neighbor interpolation."""

    tensor = torch.from_numpy(mask.astype(np.float32))
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="nearest",
    )

    return tensor.squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BraTS2DSegmentationDataset(Dataset):
    """
    Patient-aware 2D BraTS dataset.

    Four MRI modalities are used:

        1. T1
        2. T1-contrast / T1ce
        3. T2
        4. FLAIR

    The BraTS segmentation is converted to a binary tumor mask:

        0 -> background
        non-zero -> tumor

    This prototype intentionally performs 2D axial slice segmentation while
    retaining the multimodal MRI channels.
    """

    def __init__(
        self,
        patient_dirs: Sequence[Path],
        image_size: int = 224,
        min_tumor_pixels: int = 1,
    ) -> None:
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []

        for patient_dir in patient_dirs:
            segmentation_file = find_file(patient_dir, MASK_SUFFIXES)

            if segmentation_file is None:
                continue

            segmentation = load_nifti(segmentation_file)

            # A slice is included when it contains at least the requested
            # number of tumor voxels.
            for slice_index in range(segmentation.shape[2]):
                tumor_pixels = np.count_nonzero(
                    segmentation[:, :, slice_index] > 0
                )

                if tumor_pixels >= min_tumor_pixels:
                    self.samples.append((patient_dir, slice_index))

        if not self.samples:
            raise RuntimeError(
                "No valid BraTS tumor-containing slices were found. "
                "Check the dataset path and BraTS directory structure."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int,) -> tuple[torch.Tensor, torch.Tensor]:
        patient_dir, slice_index = self.samples[index]

        modality_files = find_patient_modalities(
            patient_dir
        )

        segmentation_file = find_file(
            patient_dir,
            MASK_SUFFIXES,
        )

        if (
            not all(
                path is not None
                for path in modality_files
            )
            or segmentation_file is None
        ):
            raise RuntimeError(
                f"Incomplete BraTS sample encountered: {patient_dir}"
            )

        # Make sure every required modality is a real file.
        if not all(
            path is not None and path.is_file()
            for path in modality_files
        ):
            raise RuntimeError(
                f"Missing BraTS modality file: {patient_dir}"
            )

        # Load and normalize the four MRI modalities.
        modality_slices: list[torch.Tensor] = []

        for modality_path in modality_files:
            assert modality_path is not None

            volume = load_nifti(
                modality_path
            )

            volume = normalize_modality(
                volume
            )

            slice_2d = volume[
                :,
                :,
                slice_index,
            ]

            modality_slices.append(
                resize_slice(
                    slice_2d,
                    self.image_size,
                )
            )

        image = torch.stack(
            modality_slices,
            dim=0,
        )

        # Real BraTS segmentation annotation.
        segmentation = load_nifti(
            segmentation_file
        )

        mask = (
            segmentation[
                :,
                :,
                slice_index,
            ] > 0
        ).astype(np.float32)

        mask_tensor = resize_mask(
            mask,
            self.image_size,
        )

        return (
            image.float(),
            mask_tensor.float(),
        )

# ---------------------------------------------------------------------------
# Patient-level split
# ---------------------------------------------------------------------------


def split_patients(
    patients: Sequence[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:

    if not patients:
        raise RuntimeError("No BraTS patients were discovered.")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    patients = list(patients)

    rng = random.Random(seed)
    rng.shuffle(patients)

    validation_count = max(
        1,
        int(round(len(patients) * validation_fraction)),
    )

    if validation_count >= len(patients):
        validation_count = len(patients) - 1

    validation_patients = patients[:validation_count]
    training_patients = patients[validation_count:]

    return training_patients, validation_patients


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def binary_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:

    pred = prediction >= threshold
    true = target >= 0.5

    intersection = (pred & true).sum().float()
    pred_sum = pred.sum().float()
    true_sum = true.sum().float()

    union = (pred | true).sum().float()

    dice_denominator = pred_sum + true_sum

    dice = (
        (2.0 * intersection / dice_denominator).item()
        if dice_denominator > 0
        else 1.0
    )

    iou = (
        (intersection / union).item()
        if union > 0
        else 1.0
    )

    true_positive = intersection
    false_positive = (pred & ~true).sum().float()
    false_negative = (~pred & true).sum().float()

    precision_denominator = true_positive + false_positive
    sensitivity_denominator = true_positive + false_negative

    precision = (
        (true_positive / precision_denominator).item()
        if precision_denominator > 0
        else 0.0
    )

    sensitivity = (
        (true_positive / sensitivity_denominator).item()
        if sensitivity_denominator > 0
        else 0.0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    generator.eval()

    totals = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "sensitivity": 0.0,
    }

    batch_count = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        predictions = generator(images)

        metrics = binary_segmentation_metrics(
            predictions,
            masks,
        )

        for key in totals:
            totals[key] += metrics[key]

        batch_count += 1

    if batch_count == 0:
        return totals

    return {
        key: value / batch_count
        for key, value in totals.items()
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gan(
    data_dir: Path,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 0.0002,
    validation_fraction: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
) -> None:

    set_seed(seed)

    print("=" * 72)
    print("NeuroScan-AI | ARMT-GAN Prototype Training")
    print("=" * 72)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[*] Device: {device}")
    print(f"[*] Dataset: {data_dir}")
    print(f"[*] Seed: {seed}")

    # -----------------------------------------------------------------------
    # Discover patients
    # -----------------------------------------------------------------------

    patients = discover_patients(data_dir)

    print(f"[*] Complete BraTS patients discovered: {len(patients)}")

    if len(patients) < 2:
        raise RuntimeError(
            "At least two complete BraTS patients are required "
            "for a train/validation split."
        )

    train_patients, validation_patients = split_patients(
        patients=patients,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    print(f"[*] Training patients: {len(train_patients)}")
    print(f"[*] Validation patients: {len(validation_patients)}")

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------

    train_dataset = BraTS2DSegmentationDataset(
        patient_dirs=train_patients,
        image_size=image_size,
    )

    validation_dataset = BraTS2DSegmentationDataset(
        patient_dirs=validation_patients,
        image_size=image_size,
    )

    print(f"[*] Training slices: {len(train_dataset)}")
    print(f"[*] Validation slices: {len(validation_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    generator = ARMTGenerator2D(
        in_channels=4,
        out_channels=1,
    ).to(device)

    discriminator = ARMTDiscriminator2D(
        in_channels=5,
    ).to(device)

    # -----------------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------------

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    best_dice = -1.0

    weights_dir = PROJECT_ROOT / "ai_pipeline" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0

        for batch_index, (images, real_masks) in enumerate(train_loader):
            images = images.to(device)
            real_masks = real_masks.to(device)

            # ===============================================================
            # Discriminator
            # ===============================================================

            optimizer_d.zero_grad()

            fake_masks = generator(images)

            real_prediction = discriminator(
                images,
                real_masks,
            )

            fake_prediction = discriminator(
                images,
                fake_masks.detach(),
            )

            loss_d_real = criterion_bce(
                real_prediction,
                torch.ones_like(real_prediction),
            )

            loss_d_fake = criterion_bce(
                fake_prediction,
                torch.zeros_like(fake_prediction),
            )

            loss_d = (
                loss_d_real + loss_d_fake
            ) / 2.0

            loss_d.backward()
            optimizer_d.step()

            # ===============================================================
            # Generator
            # ===============================================================

            optimizer_g.zero_grad()

            fake_masks = generator(images)

            fake_prediction = discriminator(
                images,
                fake_masks,
            )

            loss_g_adv = criterion_bce(
                fake_prediction,
                torch.ones_like(fake_prediction),
            )

            loss_g_seg = criterion_l1(
                fake_masks,
                real_masks,
            )

            # The segmentation reconstruction term remains the primary
            # prototype objective. The adversarial term is auxiliary.
            loss_g = loss_g_seg * 100.0 + loss_g_adv

            loss_g.backward()
            optimizer_g.step()

            epoch_generator_loss += loss_g.item()
            epoch_discriminator_loss += loss_d.item()

            print(
                f"[Epoch {epoch + 1}/{epochs}] "
                f"[Batch {batch_index + 1}/{len(train_loader)}] "
                f"| D: {loss_d.item():.4f} "
                f"| G: {loss_g.item():.4f}"
            )

        average_g = (
            epoch_generator_loss / max(len(train_loader), 1)
        )

        average_d = (
            epoch_discriminator_loss / max(len(train_loader), 1)
        )

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------

        validation_metrics = evaluate(
            generator=generator,
            dataloader=validation_loader,
            device=device,
        )

        print()
        print(f"--- Epoch {epoch + 1} Summary ---")
        print(f"Generator loss : {average_g:.6f}")
        print(f"Discriminator loss : {average_d:.6f}")
        print(f"Validation Dice : {validation_metrics['dice']:.6f}")
        print(f"Validation IoU : {validation_metrics['iou']:.6f}")
        print(
            f"Validation Precision : "
            f"{validation_metrics['precision']:.6f}"
        )
        print(
            f"Validation Sensitivity : "
            f"{validation_metrics['sensitivity']:.6f}"
        )

        # -------------------------------------------------------------------
        # Save latest checkpoint
        # -------------------------------------------------------------------

        latest_path = weights_dir / "generator_latest.pth"

        torch.save(
            {
                "model_state_dict": generator.state_dict(),
                "epoch": epoch + 1,
                "seed": seed,
                "image_size": image_size,
                "in_channels": 4,
                "out_channels": 1,
                "validation_metrics": validation_metrics,
            },
            latest_path,
        )

        # -------------------------------------------------------------------
        # Save best checkpoint
        # -------------------------------------------------------------------

        if validation_metrics["dice"] > best_dice:

            best_dice = validation_metrics["dice"]

            best_path = weights_dir / "generator_best.pth"

            torch.save(
                {
                    "model_state_dict": generator.state_dict(),
                    "epoch": epoch + 1,
                    "seed": seed,
                    "image_size": image_size,
                    "in_channels": 4,
                    "out_channels": 1,
                    "validation_metrics": validation_metrics,
                },
                best_path,
            )

            print(
                f"[*] New best checkpoint saved: {best_path}"
            )

        print()

    print("=" * 72)
    print("Training complete.")
    print(f"Best validation Dice: {best_dice:.6f}")
    print(f"Latest checkpoint: {weights_dir / 'generator_latest.pth'}")
    print(f"Best checkpoint: {weights_dir / 'generator_best.pth'}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the NeuroScan-AI ARMT-GAN prototype."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path containing BraTS patient directories.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    train_gan(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()