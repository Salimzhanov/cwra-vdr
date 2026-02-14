import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiple_testing import bh_qvalues


def test_bh_qvalues_toy():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # Sorted p: 0.01, 0.03, 0.04, 0.2
    # q: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.0533, 0.2*4/4=0.2
    # monotone from largest to smallest: [0.04, 0.0533, 0.0533, 0.2] in sorted order
    expected = np.array([0.04, 0.05333333, 0.05333333, 0.2])
    q = bh_qvalues(p)
    assert np.allclose(q, expected, atol=1e-6)


def test_monotonicity_sorted():
    rng = np.random.RandomState(0)
    p = rng.rand(20)
    q = bh_qvalues(p)
    order = np.argsort(p)
    q_sorted = q[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)
