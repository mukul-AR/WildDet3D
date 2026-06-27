import torch

from wilddet3d.dense.matching import match_predicted_to_gt


def test_matches_within_threshold_and_drops_far_ones():
    gt = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    pred = torch.tensor(
        [
            [0.02, 0.0, 1.0],   # near gt 0
            [0.98, 0.0, 1.0],   # near gt 1
            [5.0, 5.0, 5.0],    # far -> dropped
        ]
    )
    keep, gt_idx = match_predicted_to_gt(pred, gt, thresh=0.15)
    assert keep.tolist() == [True, True, False]
    assert gt_idx[0].item() == 0 and gt_idx[1].item() == 1


def test_empty_inputs():
    gt = torch.zeros(0, 3)
    pred = torch.zeros(0, 3)
    keep, gt_idx = match_predicted_to_gt(pred, gt)
    assert keep.numel() == 0 and gt_idx.numel() == 0
    # predictions but no GT -> nothing kept
    keep2, _ = match_predicted_to_gt(torch.randn(3, 3), torch.zeros(0, 3))
    assert keep2.tolist() == [False, False, False]
