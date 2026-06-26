import torch

from wilddet3d.dense.metrics import stage2_eval_arrays, summarize_eval


def test_perfect_prediction_scores_iou_one():
    # one scene, 2 boxes, K=3 candidate dims; make predictions exactly match GT.
    cat = torch.tensor([[0.30, 0.30, 0.50], [0.20, 0.30, 0.40], [0.25, 0.25, 0.25]])
    gt_assign = torch.tensor([0, 1])
    act_size = cat[gt_assign]
    vis_center = torch.tensor([[0.1, 0.0, 1.0], [0.2, 0.1, 1.5]])
    act_center = torch.tensor([[0.1, 0.0, 1.2], [0.2, 0.1, 1.8]])
    act_rot6d = torch.tensor([[1.0, 0, 0, 0, 1, 0]] * 2)

    logits = torch.full((1, 2, 3), -10.0)
    logits[0, 0, 0] = 10.0  # argmax -> 0
    logits[0, 1, 1] = 10.0  # argmax -> 1
    out = {
        "assign_logits": logits,
        "center_delta": (act_center - vis_center)[None],  # [1,2,3]
        "rot6d": act_rot6d[None],
        "q_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    batch = {
        "catalog": [cat],
        "vis_center": [vis_center],
        "act_center": [act_center],
        "act_size": [act_size],
        "act_rot6d": [act_rot6d],
        "assign": [gt_assign],
    }
    torch.manual_seed(0)
    arr = stage2_eval_arrays(out, batch, n_samples=20000)
    s = summarize_eval(arr)
    assert s["assign_acc"] == 1.0
    assert s["iou3d"] > 0.97
    assert s["center_dist"] < 1e-3
    assert s["size_err"] < 1e-3
    assert s["n_boxes"] == 2
