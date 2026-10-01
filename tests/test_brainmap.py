"""robot/brainmap.py on a tiny made-up connectome."""
import io

import numpy as np
import pandas as pd
import scipy.sparse as sp

from robot.brainmap import SYSTEMS, BrainActivity, top_k_rows


class C:
    def __init__(self):
        self.neurons = pd.DataFrame({
            "super_class": ["ol_intrinsic", "visual_projection", "descending_neuron", "cb_intrinsic", "vnc_motor"],
            "class": ["optic_lobe_intrinsic", "", "", "Kenyon_Cell", ""],
            "primary_type": ["Mi1", "LC10a", "DNa02", "KCab", "MN1"],
            "pos_x_nm": [1e5, 2e5, 3e5, np.nan, 5e5], "pos_y_nm": [1e5, 2e5, 3e5, np.nan, 5e5],
            "pos_z_nm": [1e5, 2e5, 3e5, np.nan, 9e5]})
        # pre -> post synapse counts (- inhibitory)
        self.w = sp.csr_matrix(np.array([[0, 5, 0, 0, 0], [0, 0, 9, -3, 1], [0, 0, 0, 0, 7], [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]], np.int32))


def test_systems_positions_and_wiring():
    bm = BrainActivity(C())
    names = [s for s, _ in SYSTEMS]
    assert [names[k] for k in bm.system] == ["Optic lobes", "Camera & lidar inputs (LC4, LPLC2, LC10a)",
                                            "Descending commands", "Learning (mushroom body)", "Motor neurons"]
    assert list(bm.idx) == [0, 1, 2, 4]                     # no position: not drawn
    idx, val = top_k_rows(C().w, 2)
    assert list(idx[1]) == [2, 3] and list(val[1]) == [9, -3]   # strongest by |synapses|
    bm.update(np.zeros(5, np.int64))
    import time
    time.sleep(0.06)
    systems, levels = bm.update(np.array([0, 20, 0, 0, 0]))
    assert systems["systems"][0]["hz"] > 0 and len(levels) == 2
    pairs = np.frombuffer(bm.flow, np.int32).reshape(-1, 2)
    assert [tuple(map(int, p)) for p in pairs] == [(1, 2), (1, 3)]   # LC10a -> DNa02, MN1 (KCab: no position)
    z = np.load(io.BytesIO(bm.circuit()), allow_pickle=False)
    assert z["types"][z["type_id"][1]] == "LC10a" and z["out_idx"][1][0] == 2
