import torch

from wilddet3d.dense.head import DenseConvHead
from wilddet3d.dense.model import DenseDet3D


def test_forward_can_return_fused_feat(monkeypatch):
    # Build a DenseDet3D with stub encoders to test the return_feat plumbing only.
    m = DenseDet3D.__new__(DenseDet3D)
    torch.nn.Module.__init__(m)
    m.fpn_level = 0
    m.head = DenseConvHead(in_ch=8)

    def fake_fused(images, depth, k):
        return [torch.randn(images.shape[0], 8, 9, 9)]

    monkeypatch.setattr(m, "_fused_feats", fake_fused, raising=False)
    out = m.forward(
        torch.randn(1, 3, 36, 36),
        torch.randn(1, 1, 36, 36),
        torch.eye(3)[None],
        return_feat=True,
    )
    assert out["feat"].shape == (1, 8, 9, 9)
    assert "heatmap" in out and "reg" in out
