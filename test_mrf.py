import itertools
import unittest
import numpy as np
from mrf import energy, delta_energy, restore, graph_cut


class MRFTests(unittest.TestCase):
    def test_delta_matches_full_energy(self):
        rng = np.random.default_rng(7)
        x = rng.choice([-1., 1.], (4, 5))
        y = rng.choice([-1., 1.], (4, 5))
        for i, j in np.ndindex(x.shape):
            flipped = x.copy(); flipped[i, j] *= -1
            self.assertAlmostEqual(energy(flipped, y, .3)-energy(x, y, .3), delta_energy(x, y, i, j, .3))

    def test_graph_cut_matches_exhaustive_minimum(self):
        for labels in itertools.product([-1., 1.], repeat=4):
            y = np.array(labels).reshape(2, 2)
            for h in [-.3, 0., .3]:
                expected = min(energy(np.array(x).reshape(2, 2), y, h) for x in itertools.product([-1., 1.], repeat=4))
                self.assertAlmostEqual(energy(graph_cut(y, h), y, h), expected)

    def test_icm_is_monotone_and_reproducible(self):
        y = np.random.default_rng(7).choice([-1., 1.], (12, 12))
        original = y.copy()
        x, info = restore(y, restarts=3)
        z, repeated = restore(y, restarts=3)
        np.testing.assert_array_equal(x, z)
        np.testing.assert_array_equal(y, original)
        self.assertEqual(info, repeated)
        for trace in info['traces']:
            self.assertTrue(np.all(np.diff(trace) <= 1e-10))
        self.assertLessEqual(energy(graph_cut(y), y), energy(x, y)+1e-10)

    def test_no_smoothness_keeps_observation(self):
        y = np.array([[1., -1.], [-1., 1.]])
        np.testing.assert_array_equal(graph_cut(y, beta=0), y)

    def test_invalid_inputs(self):
        for y in [np.array([1., -1.]), np.array([[0.]]), np.ones((0, 2))]:
            with self.assertRaises(ValueError):
                restore(y)
        with self.assertRaises(ValueError):
            graph_cut(np.ones((2, 2)), beta=-1)


if __name__ == '__main__':
    unittest.main()
