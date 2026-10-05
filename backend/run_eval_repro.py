import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

# Run evaluation twice and compare results
def run_eval():
    from ai_pipeline.evaluation.evaluate_armt_gan_baseline import (
        discover_complete_patients,
        split_patients_three_way,
        load_checkpoint,
        evaluate,
        build_result,
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

    all_patients = discover_complete_patients(data_dir)

    training_patients, validation_patients, test_patients = split_patients_three_way(
        patients=all_patients,
        train_fraction=protocol.train_fraction,
        validation_fraction=protocol.validation_fraction,
        test_fraction=protocol.test_fraction,
        seed=protocol.seed,
    )

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

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
        patients=test_patients,
        device=device,
        protocol=protocol,
        output_dir=None,
    )

    return {
        'slice_dice': slice_dice,
        'slice_iou': slice_iou,
        'slice_precision': slice_precision,
        'slice_sensitivity': slice_sensitivity,
        'patient_metrics': patient_metrics,
    }

print("RUN 1")
r1 = run_eval()
print(f"  slice_dice: {r1['slice_dice']}")
print(f"  patient_metrics: {r1['patient_metrics']}")

print("\nRUN 2")
r2 = run_eval()
print(f"  slice_dice: {r2['slice_dice']}")
print(f"  patient_metrics: {r2['patient_metrics']}")

print("\nREPRODUCIBILITY CHECK")
import numpy as np
match = True
for k in ['slice_dice', 'slice_iou', 'slice_precision', 'slice_sensitivity']:
    arr1 = np.array(r1[k])
    arr2 = np.array(r2[k])
    if not np.allclose(arr1, arr2):
        print(f"  MISMATCH in {k}")
        match = False

if match:
    print("SUCCESS: Both evaluations produce IDENTICAL results!")
else:
    print("FAILURE: Evaluations differ!")