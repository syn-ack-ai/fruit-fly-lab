"""brain/neurons/registry.trim_connectome (FLY_TRIM=robot): the robot's brain
without the optic lobes and the nerve cord."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from brain.neurons.registry import TRIM_DROP, Connectome, own_indices, trim_connectome


def tiny():
    sc = ["ol_intrinsic", "visual_projection", "cb_intrinsic", "vnc_intrinsic", "descending_neuron",
          "ascending_neuron", "vnc_motor"]
    n = pd.DataFrame({"root_id": np.arange(100, 107), "idx": np.arange(7), "super_class": sc,
                      "primary_type": ["Mi1", "LC10a", "KCab", "IN1", "DNg100", "AN1", "MN1"]})
    w = np.zeros((7, 7), np.int32)
    w[0, 1] = 5      # optic lobe -> LC10a (dropped with the optic lobe)
    w[1, 2] = 7      # LC10a -> central brain
    w[2, 4] = -3     # central -> DN
    w[4, 3] = 9      # DN -> nerve cord (dropped)
    w[5, 2] = 4      # ascending -> central
    return Connectome(n, sp.csr_matrix(w), {"dataset": "tiny", "version": "0"})


def test_trim_keeps_order_and_wiring():
    c = trim_connectome(tiny(), "robot")
    assert list(c.neurons["primary_type"]) == ["LC10a", "KCab", "DNg100", "AN1"]
    assert list(c.neurons["idx"]) == [0, 1, 2, 3] and c.n == 4
    d = c.w.toarray()
    assert d[0, 1] == 7 and d[1, 2] == -3 and d[3, 1] == 4 and np.count_nonzero(d) == 3
    assert c.idx(102) == 1                                # lookups follow
    assert c.manifest["trim"] == "robot" and c.manifest["trim_dropped"] == 3
    assert set(TRIM_DROP["robot"]) >= {"ol_intrinsic", "vnc_intrinsic", "vnc_motor"}


def test_own_indices_translate_saved_full_indices():
    full = tiny()
    c = trim_connectome(full, "robot")
    np.testing.assert_array_equal(own_indices(c, [0, 1, 4, 6]), [0, 2])   # Mi1, MN1 left out
    np.testing.assert_array_equal(own_indices(full, [0, 6]), [0, 6])      # a whole brain: unchanged
