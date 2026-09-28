"""FAFB build output must not depend on the dataset selected for simulation."""
import json

import numpy as np
import pandas as pd
import pytest

import config
from brain.connectivity import build_connectome
from brain.neurons.registry import load_connectome


@pytest.mark.parametrize("dataset", ["fafb", "malecns", "merged"])
def test_fafb_builder_preserves_other_datasets(tmp_path, monkeypatch, dataset):
    # Tiny I/O fixtures: this tests output routing, not biological connectivity.
    fafb = [tmp_path / "connectome_v783.npz", tmp_path / "neuron_index_v783.csv.gz",
            tmp_path / "build_manifest.json"]
    other = [tmp_path / dataset / name for name in ("connectome.npz", "neurons.csv.gz", "manifest.json")]
    for path in other:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"existing dataset")
    for name, path in zip(("CONNECTOME_NPZ", "NEURON_INDEX", "BUILD_MANIFEST"),
                          fafb if dataset == "fafb" else other):
        monkeypatch.setattr(config, name, path)
    for name, path in zip(("FAFB_CONNECTOME_NPZ", "FAFB_NEURON_INDEX", "FAFB_BUILD_MANIFEST"), fafb):
        monkeypatch.setattr(config, name, path, raising=False)
    monkeypatch.setattr(config, "DATASET_KEY", dataset)
    neurons = pd.DataFrame({"idx": [0, 1], "root_id": [1, 2], "nt_type": ["ACH", "GABA"]})
    connections = pd.DataFrame({"pre_root_id": [1], "post_root_id": [2],
                                "syn_count": [5], "nt_type": ["ACH"], "neuropil": ["test"]})
    monkeypatch.setattr(build_connectome, "load_neuron_table", lambda: neurons.copy())
    monkeypatch.setattr(build_connectome, "load_connections",
                        lambda _: (np.array([0]), np.array([1]), np.array([5]), connections))

    assert build_connectome.main() == 0

    assert all(path.read_bytes() == b"existing dataset" for path in other)
    with np.load(fafb[0]) as saved:
        assert saved["data"].tolist() == [5]
    assert pd.read_csv(fafb[1])["root_id"].tolist() == [1, 2]
    assert json.loads(fafb[2].read_text())["dataset"] == "FlyWire FAFB"


@pytest.mark.parametrize("dataset,builder", [
    ("fafb", "brain.connectivity.build_connectome"),
    ("malecns", "brain.connectivity.build_malecns"),
    ("merged", "brain.connectivity.merge"),
])
def test_missing_connectome_names_the_selected_builder(tmp_path, monkeypatch, dataset, builder):
    monkeypatch.setattr(config, "DATASET_KEY", dataset)
    monkeypatch.setattr(config, "CONNECTOME_NPZ", tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError, match=builder):
        load_connectome.__wrapped__()
