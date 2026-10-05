import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

from ai_pipeline.evaluation.evaluate_armt_gan_baseline import (
    discover_complete_patients,
    split_patients_three_way,
    load_checkpoint,
    evaluate,
    build_result,
    print_result,
    EvaluationProtocol,
)
import torch
from pathlib import Path

data_dir = Path(r"C:\Users\gagan\AppData\Local\Temp\synthetic_brats_hkk22lfk")
checkpoint_path = Path(r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend\ai_pipeline\weights\generator_best.pth")

protocol = EvaluationProtocol(
    train_fraction=0.5,
    validation_fraction=0.25,
    test_fraction=0.25,
)
device = torch.device("cpu")

print("Discovering patients...")
all_patients = discover_complete_patients(data_dir)

training_patients, validation_patients, test_patients = split_patients_three_way(
    patients=all_patients,
    train_fraction=protocol.train_fraction,
    validation_fraction=protocol.validation_fraction,
    test_fraction=protocol.test_fraction,
    seed=protocol.seed,
)

print(f"Complete labeled patients: {len(all_patients)}")
print(f"Training split: {len(training_patients)}")
print(f"Validation split: {len(validation_patients)}")
print(f"Test split: {len(test_patients)}")

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
    raw_artifacts,
) = evaluate(
    model=model,
    patients=test_patients,  # Evaluate on TEST split
    device=device,
    protocol=protocol,
    output_dir=None,
)

result = build_result(
    checkpoint_path=checkpoint_path,
    device=device,
    protocol=protocol,
    all_patients=all_patients,
    training_patients=training_patients,
    validation_patients=validation_patients,
    test_patients=test_patients,
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
    raw_artifacts=raw_artifacts,
)

print_result(result)