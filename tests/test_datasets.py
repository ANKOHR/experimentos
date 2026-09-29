import hashlib

from experimentos.datasets import list_datasets, register_dataset


def test_registration_records_schema_checksum_seed_and_generation(home):
    entry = list_datasets(home)[0]
    csv_bytes = (home / entry["path"]).read_bytes()
    assert entry["schema"] == ["sample", "feature_1", "feature_2", "target"]
    assert entry["row_count"] == 12
    assert entry["generation"]["seed"] == 19
    assert entry["generation"]["features"] == 2
    assert entry["sha256"] == hashlib.sha256(csv_bytes).hexdigest()


def test_reregister_same_seed_produces_identical_file_and_checksum(home):
    before = (home / "datasets" / "toy.csv").read_bytes()
    first = register_dataset("toy", seed=19, rows=12, features=2, root=home)
    after = (home / "datasets" / "toy.csv").read_bytes()
    second = register_dataset("toy", seed=19, rows=12, features=2, root=home)
    assert before == after
    assert first["sha256"] == second["sha256"]


def test_dataset_csv_changes_when_seed_changes(home):
    original = (home / "datasets" / "toy.csv").read_bytes()
    register_dataset("toy", seed=20, rows=12, features=2, root=home)
    assert (home / "datasets" / "toy.csv").read_bytes() != original
