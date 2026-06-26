import torch

from wilddet3d.dense.decode import decode_jenga
from wilddet3d.dense.stage2 import JengaStage2


def test_decode_selects_catalog_dim_and_completes_pose():
    s2 = JengaStage2(in_ch=8, d_model=32, layers=2, heads=4).eval()
    feat = torch.randn(1, 8, 12, 12)
    dets = [
        {
            "center": torch.tensor([[0.1, 0.0, 1.0], [0.0, 0.1, 2.0]]),
            "size": torch.rand(2, 3),
            "R": torch.eye(3).expand(2, 3, 3),
            "score": torch.tensor([0.9, 0.8]),
        }
    ]
    catalog = [torch.tensor([[0.3, 0.34, 0.49], [0.3, 0.4, 0.4]])]
    k = torch.tensor([[[500.0, 0, 6.0], [0, 500.0, 6.0], [0, 0, 1.0]]])
    out = decode_jenga(dets, feat, stride=84.0, stage2=s2, catalog=catalog, k=k)
    o = out[0]
    assert o["size"].shape == (2, 3) and o["center"].shape == (2, 3)
    assert o["R"].shape == (2, 3, 3)
    # selected size must be an exact catalog row
    for s in o["size"]:
        assert torch.any(torch.all(torch.isclose(catalog[0], s, atol=1e-5), dim=1))
