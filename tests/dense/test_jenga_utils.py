import numpy as np

from wilddet3d.dense.jenga_utils import assign_index, canonicalize_obb


def test_canonicalize_sorts_size_and_keeps_proper_rotation():
    R = np.eye(3)
    size = np.array([0.49, 0.34, 0.34])  # unsorted
    s2, R2 = canonicalize_obb(size, R)
    assert np.allclose(s2, [0.34, 0.34, 0.49])  # ascending
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-5)  # still a rotation
    # column that had the largest extent (x) must now be last
    assert np.allclose(R2[:, 2], R[:, 0])


def test_canonicalize_fixes_handedness_on_odd_permutation():
    R = np.eye(3)
    size = np.array([0.3, 0.5, 0.4])  # argsort -> [0,2,1] is odd
    _, R2 = canonicalize_obb(size, R)
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-5)


def test_assign_index_picks_nearest_catalog_row():
    cat = np.array([[0.34, 0.34, 0.49], [0.30, 0.40, 0.40]])
    assert assign_index(np.array([0.34, 0.34, 0.49]), cat) == 0
    assert assign_index(np.array([0.31, 0.39, 0.41]), cat) == 1
