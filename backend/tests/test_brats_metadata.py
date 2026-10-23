
import csv

from app.services.brats_metadata import resolve_brats2020_cohort


def write_mapping(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["Grade", "BraTS_2020_subject_ID"]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_resolves_verified_hgg_subject(tmp_path):
    mapping = tmp_path / "name_mapping.csv"
    write_mapping(
        mapping,
        [{"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_001"}],
    )

    result = resolve_brats2020_cohort("BraTS20_Training_001", mapping)

    assert result == {
        "subject_id": "BraTS20_Training_001",
        "cohort_label": "HGG",
        "source": "BraTS2020_TrainingData/name_mapping.csv",
    }


def test_resolves_verified_lgg_subject(tmp_path):
    mapping = tmp_path / "name_mapping.csv"
    write_mapping(
        mapping,
        [{"Grade": "LGG", "BraTS_2020_subject_ID": "BraTS20_Training_002"}],
    )

    result = resolve_brats2020_cohort("BraTS20_Training_002", mapping)

    assert result is not None
    assert result["cohort_label"] == "LGG"


def test_unknown_subject_returns_none(tmp_path):
    mapping = tmp_path / "name_mapping.csv"
    write_mapping(
        mapping,
        [{"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_001"}],
    )

    assert resolve_brats2020_cohort("unknown-subject", mapping) is None


def test_missing_or_blank_subject_returns_none(tmp_path):
    mapping = tmp_path / "name_mapping.csv"
    write_mapping(mapping, [])

    assert resolve_brats2020_cohort(None, mapping) is None
    assert resolve_brats2020_cohort("", mapping) is None
    assert resolve_brats2020_cohort("   ", mapping) is None


def test_unrecognized_cohort_label_returns_none(tmp_path):
    mapping = tmp_path / "name_mapping.csv"
    write_mapping(
        mapping,
        [{"Grade": "UNKNOWN", "BraTS_2020_subject_ID": "BraTS20_Training_001"}],
    )

    assert resolve_brats2020_cohort("BraTS20_Training_001", mapping) is None


def test_missing_mapping_file_returns_none(tmp_path):
    assert (
        resolve_brats2020_cohort(
            "BraTS20_Training_001", tmp_path / "missing.csv"
        )
        is None
    )


def test_missing_required_columns_returns_none(tmp_path):
    mapping = tmp_path / "wrong_columns.csv"
    write_mapping(
        mapping,
        [{"grade": "HGG", "subject": "BraTS20_Training_001"}],
        fieldnames=["grade", "subject"],
    )

    assert resolve_brats2020_cohort("BraTS20_Training_001", mapping) is None