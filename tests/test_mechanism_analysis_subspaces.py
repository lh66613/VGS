import numpy as np
import torch

from scripts.dump_mechanism_mitigation_operator_geometry import _subspace_bases


def test_dynamic_band_and_single_direction_projectors():
    vh = np.eye(20, dtype=np.float32)
    projectors = _subspace_bases(vh, ["band1_12", "band9_20", "v5"], seed=42)
    x = torch.arange(1, 21, dtype=torch.float32)

    band1_12 = projectors["band1_12"](x)
    assert torch.allclose(band1_12[:12], x[:12])
    assert torch.count_nonzero(band1_12[12:]) == 0

    band9_20 = projectors["band9_20"](x)
    assert torch.count_nonzero(band9_20[:8]) == 0
    assert torch.allclose(band9_20[8:], x[8:])

    v5 = projectors["v5"](x)
    assert v5[4].item() == x[4].item()
    assert torch.count_nonzero(torch.cat([v5[:4], v5[5:]])) == 0


def test_leave_one_out_and_random_contiguous_projectors_are_valid():
    vh = np.eye(32, dtype=np.float32)
    projectors = _subspace_bases(vh, ["band5_16_minus_v5", "randcontig12_s00"], seed=42)
    x = torch.ones(32, dtype=torch.float32)

    leave_one_out = projectors["band5_16_minus_v5"](x)
    assert leave_one_out[4].item() == 0
    assert torch.count_nonzero(leave_one_out[5:16]).item() == 11

    random_contiguous = projectors["randcontig12_s00"](x)
    assert torch.count_nonzero(random_contiguous).item() == 12
