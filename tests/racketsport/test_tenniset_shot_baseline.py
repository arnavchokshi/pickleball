import numpy as np

from scripts.racketsport.train_tenniset_shot_baseline import (
    build_dataset_index,
    load_feature_vector,
    macro_f1,
)


def _write_sample(root, split, label, sample_id, offset):
    sample_dir = root / split / label
    sample_dir.mkdir(parents=True, exist_ok=True)
    joints = np.full((3, 2, 17, 2), offset, dtype=np.float32)
    pos = np.full((3, 2, 2), offset + 1.0, dtype=np.float32)
    shuttle = np.full((3, 2), offset + 2.0, dtype=np.float32)
    np.save(sample_dir / f"{sample_id}_joints.npy", joints)
    np.save(sample_dir / f"{sample_id}_pos.npy", pos)
    np.save(sample_dir / f"{sample_id}_shuttle.npy", shuttle)


def test_build_dataset_index_finds_complete_tenniset_samples(tmp_path):
    _write_sample(tmp_path, "Train", "HFL", "1_001", 0.1)
    _write_sample(tmp_path, "Train", "HFR", "2_001", 0.2)
    incomplete = tmp_path / "Train" / "HFR" / "broken_joints.npy"
    np.save(incomplete, np.zeros((3, 2, 17, 2), dtype=np.float32))

    samples, labels = build_dataset_index(tmp_path, "Train")

    assert labels == ["HFL", "HFR"]
    assert [(sample.sample_id, sample.label) for sample in samples] == [
        ("1_001", "HFL"),
        ("2_001", "HFR"),
    ]


def test_load_feature_vector_resamples_and_flattens_modalities(tmp_path):
    _write_sample(tmp_path, "Train", "HFL", "1_001", 0.5)
    sample = build_dataset_index(tmp_path, "Train")[0][0]

    feature = load_feature_vector(sample, seq_len=5)

    assert feature.shape == (5 * (2 * 17 * 2 + 2 * 2 + 2),)
    assert np.isfinite(feature).all()
    assert feature[0] == 0.5


def test_macro_f1_penalizes_missed_classes():
    assert macro_f1([0, 1, 1], [0, 0, 1], class_count=2) == 0.666667
