import sys
sys.path.insert(0, r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend")

import subprocess
import sys

result = subprocess.run([
    sys.executable, "-m", "ai_pipeline.evaluation.evaluate_armt_gan_baseline",
    "--data_dir", r"C:\Users\gagan\AppData\Local\Temp\synthetic_brats_hkk22lfk",
    "--checkpoint", r"C:\Users\gagan\Downloads\Major project\neuroscan-ai\backend\ai_pipeline\weights\generator_best.pth",
    "--output_dir", r"C:\Users\gagan\AppData\Local\Temp\eval_output",
    "--train_fraction", "0.5",
    "--validation_fraction", "0.25",
    "--test_fraction", "0.25",
    "--save_predictions",
    "--output_json", r"C:\Users\gagan\AppData\Local\Temp\eval_output\eval_result.json"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)