from typing import Tuple
import torch
from torch import Tensor


def bxyz2flat(in_coords: Tensor, coords_min: Tensor,
              coords_max: Tensor) -> Tensor:
    """ (((b) * X_SIZE + x) * Y_SIZE + y) * Z_SIZE + z
    Args:
        in_coords (Tensor): bxyz coordinate [L, BXYZ_DIM]

    Return
        Tensor: flat coordinate. shape[L]
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


def flat2bxyz(in_coords: Tensor, coords_min: Tensor,
              coords_max: Tensor) -> Tensor:
    """(((b) * X_SIZE + x) * Y_SIZE + y) * Z_SIZE + z
    Args:
        in_coords (Tensor): flat coordinate. shape[L]
        coords_min (Tensor): minimum coordinate. shape[BXYZ_DIM]
        coords_max (Tensor): maximum coordinate. shape[BXYZ_DIM]

    Returns:
        Tensor: bxyz coordinate [L, BXYZ_DIM]
    """
    BXYZ_DIM = 4

    cur = in_coords.clone()
    out_coords = torch.zeros(len(in_coords),
                             BXYZ_DIM,
                             dtype=in_coords.dtype,
                             device=in_coords.device)

    sizes = coords_max - coords_min + 1

    for idx in range(BXYZ_DIM - 1, -1, -1):
        out_coords[:, idx] = coords_min[idx] + cur % sizes[idx]
        cur //= sizes[idx]

    return out_coords


def generate_rand_input(batch_size=2,
                        channel_size=2,
                        length=3) -> Tuple[Tensor, Tensor]:
    XYZ_DIM = 3
    feats_list = []
    coords_list = []

    for b_idx in range(batch_size):
        feats = torch.rand(length, channel_size)
        coords = torch.randint(0, 4, (length, XYZ_DIM))
        feats_list.append(feats)

        batch_idx = torch.full((length, 1),
                               b_idx,
                               dtype=coords.dtype,
                               device=coords.device)
        coords_list.append(torch.cat((batch_idx, coords), dim=1))

    return torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0)


def test():
    _, in_coords = generate_rand_input(batch_size=10,
                                       channel_size=8,
                                       length=100)

    coords_min, _ = torch.min(in_coords, dim=0)
    coords_max, _ = torch.max(in_coords, dim=0)

    flat_coords = bxyz2flat(in_coords, coords_min, coords_max)
    out_coords = flat2bxyz(flat_coords, coords_min, coords_max)

    success = torch.sum(in_coords != out_coords).item() == 0

    return success


def main():
    success = test()
    print("test success=", success)


if __name__ == "__main__":
    main()
