from typing import Tuple
import torch
from torch import Tensor, nn
XYZ_DIM = 3  # [X,Y,Z]


def encode_coordinate(in_coords: Tensor, coords_min: Tensor,
                      coords_max: Tensor) -> Tensor:
    """ (((b) * X_SIZE + x) * Y_SIZE + y) * Z_SIZE + z
    Args:
        in_coords (Tensor): [L, BXYZ_DIM]

    Return
        Tensor: encoded coordinate. shape[L]
        Tensor: minimum coordiante. shppe[BXYZ_DIM]
        Tensor: maximum coordinate. shape[BXYZ_DIM]
    """
    BXYZ_DIM = 4

    assert in_coords.shape[1] == BXYZ_DIM

    sizes = coords_max - coords_min + 1

    cur = torch.zeros(in_coords.shape[0],
                      dtype=in_coords.dtype,
                      device=in_coords.device)
    for i in range(BXYZ_DIM):
        cur *= sizes[i]
        cur += (in_coords[:, i] - coords_min[i])

    return cur


def decode_coordinate(in_coords: Tensor, coords_min: Tensor,
                      coords_max: Tensor) -> Tensor:
    """
    Args:
        in_coords (Tensor): encoded coordinate. shape[L]
        coords_min (Tensor): minimum coordinate. shape[BXYZ_DIM]
        coords_max (Tensor): maximum coordinate. shape[BXYZ_DIM]

    Returns:
        Tensor: decoded coordinate [L, BXYZ_DIM]
    """
    BXYZ_DIM = 4

    cur = in_coords.clone()
    out_coords = torch.zeros(len(in_coords),
                             BXYZ_DIM,
                             dtype=in_coords.dtype,
                             device=in_coords.device)

    sizes = coords_max - coords_min + 1

    # print(cur.shape)

    for idx in range(BXYZ_DIM - 1, -1, -1):
        out_coords[:, idx] = coords_min[idx] + cur % sizes[idx]
        cur //= sizes[idx]

    return out_coords


def sparse_max_pool_1d(in_feats: Tensor, in_coords: Tensor,
                       kernel_size: int) -> Tuple[Tensor, Tensor]:
    """_summary_

    Args:
        in_feats (Tensor): shape[L, C]
        in_coords (Tensor): shape[L, BXYZ_DIM]
        kernel_size (int): downsample ratio

    Returns:
        Tuple[Tensor, Tensor]: out_feats[L*, C], out_coords[L*, BXYZ_DIM]
    """
    # print("max_batch=", torch.max(in_coords[:, 0]))

    out_coords = in_coords.clone()
    out_coords[:, 1:] = in_coords[:, 1:] // kernel_size

    out_coords, _ = torch.sort(out_coords)

    coords_min, _ = torch.min(in_coords, dim=0)
    coords_max, _ = torch.max(in_coords, dim=0)

    # print("min,max=", coords_min, coords_max)

    print("before=",out_coords[-10:, :])
    enc_coords = encode_coordinate(out_coords, coords_min, coords_max)
    dec_coords = decode_coordinate(enc_coords, coords_min, coords_max)
    print("after=",dec_coords[-10:, :])
    
    enc_coords, inv_idx = torch.unique(enc_coords,
                                       sorted=False,
                                       return_inverse=True,
                                       return_counts=False,
                                       dim=0)

    out_feats = torch.empty(len(enc_coords),
                            in_feats.shape[1],
                            dtype=in_feats.dtype,
                            device=in_feats.device)

    # reduce same coordinate
    out_feats.index_reduce_(0, inv_idx, in_feats, 'amax', include_self=False)

    # print("before=",out_coords[-10:, :])

    out_coords = decode_coordinate(enc_coords, coords_min, coords_max)

    # print("after=",out_coords[-10:, :])


    return out_feats, out_coords
