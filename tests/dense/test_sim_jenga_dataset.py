import os

import pytest
import torch

from wilddet3d.dense.sim_jenga_dataset import SimJengaDataset, jenga_collate

ROOT = "/storage/3dl_sim_data/20260625_2skuwallremoval/anyware-sim/build/scenes/synth"


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_sample_has_visible_actual_catalog_and_assignment():
    ds = SimJengaDataset(ROOT, size=1008, max_scenes=4, split="train", val_frac=0.0)
    s = ds[0]
    n = s["vis_center"].shape[0]
    assert s["act_center"].shape[0] == n and s["assign"].shape[0] == n
    assert s["act_size"].shape == (n, 3) and s["act_rot6d"].shape == (n, 6)
    # actual size is stored NATIVE (per-axis); its sorted dims match the
    # assigned catalog row (the box shares the visible orientation)
    cat = s["catalog"]
    k = cat.shape[0]
    assert k >= 1 and int(s["assign"].max()) < k
    for j in range(n):
        ss = s["act_size"][j].sort().values
        assert torch.allclose(cat[s["assign"][j]], ss, atol=1e-3)


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_train_val_split_is_disjoint():
    tr = SimJengaDataset(ROOT, max_scenes=20, split="train", val_frac=0.2)
    va = SimJengaDataset(ROOT, max_scenes=20, split="val", val_frac=0.2)
    assert set(tr.samples).isdisjoint(set(va.samples))
    assert len(va) > 0 and len(tr) > 0


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="sim data not present")
def test_collate_keeps_catalog_per_sample():
    ds = SimJengaDataset(ROOT, max_scenes=4, split="train", val_frac=0.0)
    b = jenga_collate([ds[0], ds[1]])
    assert b["image"].shape[0] == 2
    assert isinstance(b["catalog"], list) and len(b["catalog"]) == 2
