import torch

from wilddet3d.dense.loss import JengaStage2Loss


def _fake(b=2, q=3, k=4):
    out = {
        "assign_logits": torch.randn(b, q, k, requires_grad=True),
        "center_delta": torch.randn(b, q, 3, requires_grad=True),
        "q_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "k_mask": torch.ones(b, k, dtype=torch.bool),
    }
    batch = {
        "vis_center": [torch.zeros(3, 3), torch.zeros(2, 3)],
        "act_center": [torch.randn(3, 3), torch.randn(2, 3)],
        "assign": [torch.tensor([0, 1, 2]), torch.tensor([3, 0])],
    }
    return out, batch


def test_loss_is_finite_and_differentiable():
    loss = JengaStage2Loss()
    out, batch = _fake()
    d = loss(out, batch)
    for k in ("assign", "center", "total"):
        assert torch.isfinite(d[k]).all()
    assert "rot" not in d  # no rotation term
    d["total"].backward()
    assert out["assign_logits"].grad is not None
    assert out["center_delta"].grad is not None
    assert 0.0 <= d["assign_acc"].item() <= 1.0
