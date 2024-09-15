from typing import Tuple
import torch
from torch import Tensor, nn
from torchsparse import SparseTensor
from flat_coord import bxyz2flat, flat2bxyz

__all__ = ["MaxPool3d"]

XYZ_DIM = 3  # [X,Y,Z]


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

    out_coords = in_coords.clone()
    out_coords[:, 1:] = in_coords[:, 1:] // kernel_size

    coords_min, _ = torch.min(out_coords, dim=0)
    coords_max, _ = torch.max(out_coords, dim=0)

    enc_coords = bxyz2flat(out_coords, coords_min, coords_max)

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

    out_coords = flat2bxyz(enc_coords, coords_min, coords_max)

    return out_feats, out_coords


class MaxPool3d(nn.Module):

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        assert isinstance(kernel_size, int), "currently support only integer"
        self.kernel_size = kernel_size

    def forward(self, input: SparseTensor) -> SparseTensor:
        out_feats, out_coords = sparse_max_pool_1d(in_feats=input.feats,
                                                   in_coords=input.coords,
                                                   kernel_size=self.kernel_size)

        output = SparseTensor(
            coords=out_coords,
            feats=out_feats,
            stride=[stride * self.kernel_size for stride in input.stride],
            spatial_range=input.spatial_range,
        )

        output._caches = input._caches

        return output


def main():
    channel_size = 2
    in_coords = torch.Tensor([[0, 1, 2, 3], [0, 1, 2, 2], [1, 1, 2, 2]])
    gt_coords = torch.Tensor([[0, 0, 1, 1], [1, 0, 1, 1]])

    in_feats = torch.rand(in_coords.shape[0], channel_size)
    print(in_feats, in_coords)
    out_feats, out_coords = sparse_max_pool_1d(in_feats=in_feats,
                                               in_coords=in_coords,
                                               kernel_size=2)
    print(out_feats, out_coords)
    success = torch.sum(gt_coords != out_coords).item() == 0
    print(success)


if __name__ == "__main__":
    main()
