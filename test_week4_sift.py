# Tom Kazakov
# Unit tests for SIFT keypoint localization math
# Auto-generated with Claude Opus 4.6

import numpy as np
import pytest

from week4_sift import gradient_and_hessian, localize_keypoint, S, CONTRAST_THRESHOLD


def _make_quadratic_dogs(center_val, grad, hessian_mat, shape=(7, 7), num_scales=5):
    """Build a synthetic DoG stack where dogs[2][3,3] sits at (s,r,c) = (2,3,3)
    and the local Taylor expansion has the given gradient and Hessian."""
    dogs = [np.zeros(shape, dtype=np.float64) for _ in range(num_scales)]
    s0, r0, c0 = 2, 3, 3
    for si in range(num_scales):
        for ri in range(shape[0]):
            for ci in range(shape[1]):
                dx = np.array([ci - c0, ri - r0, si - s0], dtype=np.float64)
                dogs[si][ri, ci] = center_val + grad @ dx + 0.5 * dx @ hessian_mat @ dx
    return dogs


class TestGradientAndHessian:
    def test_recovers_known_gradient(self):
        grad = np.array([0.1, -0.2, 0.05])
        H = np.diag([-1.0, -1.0, -1.0])
        dogs = _make_quadratic_dogs(0.5, grad, H)
        g_est, _ = gradient_and_hessian(dogs, s=2, r=3, c=3)
        np.testing.assert_allclose(g_est, grad, atol=1e-12)

    def test_recovers_known_hessian(self):
        grad = np.array([0.0, 0.0, 0.0])
        H = np.array([[-2.0, 0.3, -0.1], [0.3, -1.5, 0.2], [-0.1, 0.2, -0.8]])
        dogs = _make_quadratic_dogs(0.5, grad, H)
        _, H_est = gradient_and_hessian(dogs, s=2, r=3, c=3)
        np.testing.assert_allclose(H_est, H, atol=1e-12)

    def test_zero_gradient_at_center_of_pure_quadratic(self):
        H = np.diag([-1.0, -2.0, -0.5])
        dogs = _make_quadratic_dogs(1.0, np.zeros(3), H)
        g_est, _ = gradient_and_hessian(dogs, s=2, r=3, c=3)
        np.testing.assert_allclose(g_est, 0.0, atol=1e-12)


class TestLocalizeKeypoint:
    def test_converges_at_grid_center(self):
        """Extremum exactly at (2,3,3) — offset should be zero, contrast = center value."""
        H = np.diag([-1.0, -1.0, -1.0])
        dogs = _make_quadratic_dogs(0.5, np.zeros(3), H)
        result = localize_keypoint(dogs, s=2, r=3, c=3)
        assert result is not None
        s, r, c, contrast = result
        assert (s, r, c) == (2, 3, 3)
        assert abs(contrast - 0.5) < 1e-10

    def test_finds_offset_extremum(self):
        """Extremum at (2, 3.3, 3.2) — should converge to grid point (2,3,3) with sub-pixel offset."""
        true_offset = np.array([0.2, 0.3, 0.0])
        H = np.diag([-2.0, -2.0, -2.0])
        grad = (
            -H @ true_offset
        )  # gradient at grid point that places extremum at true_offset
        center_val = 0.8
        dogs = _make_quadratic_dogs(center_val, grad, H)
        result = localize_keypoint(dogs, s=2, r=3, c=3)
        assert result is not None
        _, _, _, contrast = result
        true_contrast = center_val + 0.5 * grad @ true_offset
        assert abs(contrast - true_contrast) < 1e-10

    def test_rejects_low_contrast(self):
        """Extremum with contrast below threshold is rejected."""
        tiny_val = (CONTRAST_THRESHOLD / S) * 0.5
        H = np.diag([-1.0, -1.0, -1.0])
        dogs = _make_quadratic_dogs(tiny_val, np.zeros(3), H)
        result = localize_keypoint(dogs, s=2, r=3, c=3)
        assert result is None

    def test_rejects_unconverged(self):
        """Strong gradient that keeps pushing the offset past 0.5 each step."""
        H = np.diag([-0.01, -0.01, -0.01])  # very weak curvature
        grad = np.array([5.0, 5.0, 5.0])  # strong gradient pulls far from center
        dogs = _make_quadratic_dogs(0.5, grad, H, shape=(50, 50), num_scales=20)
        result = localize_keypoint(dogs, s=10, r=25, c=25)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
