
"""Resolve verified BraTS 2020 dataset cohort metadata.

HGG/LGG are returned only as source-dataset cohort labels.
They are not ARMT-GAN predictions or independently validated WHO grades.
"""
import csv
from pathlib import Path
from typing import Optional



_MAPPING_PATH = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "BraTS_data"
    / "BraTS2020_TrainingData"
    / "name_mapping.csv"
)

_ALLOWED_GRADES = {"HGG", "LGG"}


def resolve_brats2020_cohort(
    subject_id: Optional[str],
    mapping_path: Optional[Path] = None,
) -> Optional[dict[str, str]]:
    """Return verified dataset cohort metadata, or None if unmatched.

    The subject ID must exactly match a BraTS 2020 training ID in the mapping.
    Missing IDs, unknown IDs, malformed mapping files, and unavailable files
    return None rather than inventing metadata.
    """
    if not isinstance(subject_id, str) or not subject_id.strip():
        return None

    path = Path(mapping_path) if mapping_path is not None else _MAPPING_PATH

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"Grade", "BraTS_2020_subject_ID"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return None

            for row in reader:
                candidate_id = (row.get("BraTS_2020_subject_ID") or "").strip()
                grade = (row.get("Grade") or "").strip().upper()

                if candidate_id == subject_id.strip():
                    if grade not in _ALLOWED_GRADES:
                        return None
                    return {
                        "subject_id": candidate_id,
                        "cohort_label": grade,
                        "source": "BraTS2020_TrainingData/name_mapping.csv",
                    }
    except (OSError, csv.Error, UnicodeError):
        return None

    return None