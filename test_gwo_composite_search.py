"""Cheap invariants for the GWO search geometry and reproducibility."""
import unittest

import numpy as np

from gwo_composite_search import coefficients, move_wolves, repair_triangle
from m import CompositeRobustLoss


class GWOTests(unittest.TestCase):
    def test_repair_produces_valid_three_loss_mixture(self):
        for raw in ([1.4, .7], [-.2, .4], [0, 0], [.424, .357]):
            with self.subTest(raw=raw):
                mixture = coefficients(raw)
                self.assertTrue(all(0 <= weight <= 1 for weight in mixture))
                self.assertAlmostEqual(sum(mixture), 1.0, places=12)
                CompositeRobustLoss(weights=mixture, task_weights=(.9, .1))

    def test_wolf_update_is_deterministic_and_feasible(self):
        population = np.asarray([[.1, .2], [.2, .3], [.3, .4], [.4, .1]])
        leaders = [{"position": row.tolist()} for row in population[:3]]
        first = move_wolves(population, leaders, 2.0, np.random.default_rng(2026))
        second = move_wolves(population, leaders, 2.0, np.random.default_rng(2026))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue((first >= 0).all())
        self.assertTrue((first.sum(axis=1) <= 1 + 1e-12).all())
        self.assertFalse(np.array_equal(first, population))

    def test_reject_nonfinite_position(self):
        with self.assertRaises(ValueError):
            repair_triangle([np.nan, .2])


if __name__ == "__main__":
    unittest.main()
