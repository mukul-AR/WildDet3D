import torch

from wilddet3d.dense.loss import JengaStage2Loss


def _fake(b=2, q=3, k=4):
    out = {
        "assign_logits": torch.randn(b, q, k, requires_grad=True),
        "face_delta": torch.randn(b, q, 3, requires_grad=True),
        "log_size": torch.randn(b, q, 3, requires_grad=True),
        "q_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "k_mask": torch.ones(b, k, dtype=torch.bool),
    }
    ident6 = lambda n: torch.tensor([[1.0, 0, 0, 0, 1, 0]] * n)
    front = lambda n: torch.rand(n, 3) * 0.2 + torch.tensor([0.0, 0.0, 1.0])
    batch = {
        "vis_center": [front(3), front(2)],
        "vis_size": [torch.rand(3, 3) + 0.2, torch.rand(2, 3) + 0.2],
        "vis_rot6d": [ident6(3), ident6(2)],
        "act_center": [torch.randn(3, 3), torch.randn(2, 3)],
        "act_size": [torch.rand(3, 3) + 0.2, torch.rand(2, 3) + 0.2],
        "act_rot6d": [ident6(3), ident6(2)],
        "assign": [torch.tensor([0, 1, 2]), torch.tensor([3, 0])],
    }
    return out, batch


def test_loss_is_finite_and_differentiable():
    loss = JengaStage2Loss()
    out, batch = _fake()
    d = loss(out, batch)
    for k in ("assign", "center", "size", "add", "total"):
        assert torch.isfinite(d[k]).all()
    assert "rot" not in d  # no rotation term
    d["total"].backward()
    assert out["assign_logits"].grad is not None
    assert out["face_delta"].grad is not None
    assert out["log_size"].grad is not None  # ADD + size terms train the per-axis head
    assert 0.0 <= d["assign_acc"].item() <= 1.0
