import torch

from wilddet3d.dense.stage2 import JengaStage2


def test_forward_shapes_and_masking_variable_q_and_k():
    m = JengaStage2(in_ch=16, d_model=32, layers=2, heads=4).eval()
    b, c, hf, wf = 2, 16, 12, 12
    feat = torch.randn(b, c, hf, wf)
    queries_uv = [torch.rand(3, 2) * (wf - 1), torch.rand(1, 2) * (wf - 1)]
    vis_obb = [torch.randn(3, 12), torch.randn(1, 12)]
    catalog = [torch.rand(2, 3), torch.rand(4, 3)]
    out = m(feat, queries_uv, vis_obb, catalog)
    assert out["assign_logits"].shape == (2, 3, 4)  # Qmax=3, Kmax=4
    assert out["center_delta"].shape == (2, 3, 3)
    assert out["log_size"].shape == (2, 3, 3)  # per-axis extents (dim->axis assignment)
    assert "rot6d" not in out  # rotation is inherited, not predicted
    assert out["q_mask"].sum().item() == 4  # 3 + 1 real queries
    # padded dim-token slots must be masked out (logit -> -inf) for sample 0
    assert torch.isinf(out["assign_logits"][0, 0, 2:]).all()


def test_backward_runs():
    m = JengaStage2(in_ch=16, d_model=32, layers=2, heads=4)
    feat = torch.randn(1, 16, 12, 12, requires_grad=True)
    out = m(feat, [torch.rand(2, 2) * 11], [torch.randn(2, 12)], [torch.rand(3, 3)])
    (out["center_delta"].sum() + out["assign_logits"].sum() + out["log_size"].sum()).backward()
    assert feat.grad is not None
