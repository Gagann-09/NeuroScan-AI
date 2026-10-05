import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

from ai_pipeline.evaluation.evaluate_armt_gan_baseline import (
    discover_complete_patients,
    split_patients,
    load_checkpoint,
    evaluate,
    build_result,
    print_result,
    EvaluationConfig,
    BASELINE_EVAL_CONFIG,
)
import torch
from pathlib import Path

data_dir = Path(r"C:\Users\gagan\AppData\Local\Temp\synthetic_brats_hkk22lfk")
checkpoint_path = Path(r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend\ai_pipeline\weights\generator_best.pth")

eval_config = BASELINE_EVAL_CONFIG
device = torch.device("cpu")

print("Discovering patients...")
all_patients = discover_complete_patients(data_dir)

training_patients, validation_patients = split_patients(
    patients=all_patients,
    validation_fraction=eval_config.validation_fraction,
    seed=eval_config.seed,
)

print(f"Complete labeled patients: {len(all_patients)}")
print(f"Training split: {len(training_patients)}")
print(f"Held-out split: {len(validation_patients)}")

model, checkpoint_metadata = load_checkpoint(
    checkpoint_path=checkpoint_path,
    device=device,
)

print(f"Checkpoint metadata: {checkpoint_metadata}")

(
    slice_dice,
    slice_iou,
    slice_precision,
    slice_sensitivity,
    patient_metrics,
    evaluated_slices,
    skipped_slices,
    total_inference_seconds,
    mean_slice_latency_ms,
) = evaluate(
    model=model,
    patients=validation_patients,
    device=device,
    image_size=eval_config.image_size,
    min_tumor_pixels=eval_config.min_tumor_pixels,
)

result = build_result(
    checkpoint_path=checkpoint_path,
    device=device,
    seed=eval_config.seed,
    image_size=eval_config.image_size,
    validation_fraction=eval_config.validation_fraction,
    all_patients=all_patients,
    training_patients=training_patients,
    validation_patients=validation_patients,
    slice_metrics=(
        slice_dice,
        slice_iou,
        slice_precision,
        slice_sensitivity,
    ),
    patient_metrics=patient_metrics,
    evaluated_slices=evaluated_slices,
    skipped_slices=skipped_slices,
    total_inference_seconds=total_inference_seconds,
    mean_slice_latency_ms=mean_slice_latency_ms,
    checkpoint_metadata=checkpoint_metadata,
)

print_result(result)