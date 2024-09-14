import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# from mmengine.runner import auto_fp16
from fp16_utils import auto_fp16
from torch.nn import functional as F
import time
# from ..builder import MIDDLE_ENCODERS
from max_pool import sparse_max_pool_1d
from torch.nn.utils.rnn import pad_sequence
__all__ = ["FlatFormer"]

def _create_cu_seqlens(batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        num_tokens * (batch_size + 1),
        step=num_tokens,
        dtype=torch.int32,
        device=device,
    )


# class FlashAttention(nn.MultiheadAttention):
#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         assert self._qkv_same_embed_dim

#         batch_size, num_tokens, embed_dim = q.shape
#         head_dim = embed_dim // self.num_heads

#         x = torch.stack([q, k, v])
#         x = x.view(3, -1, x.shape[-1])
#         x = torch.baddbmm(self.ib(), x, self.iw())
#         qkv = x.view(3, -1, self.num_heads, head_dim).transpose(0, 1)

#         cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, qkv.device)
#         x = flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, num_tokens, 0)
#         x = x.view(batch_size, num_tokens, -1)

#         x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
#         return x, None

#     def iw(self) -> torch.Tensor:
#         tensor = self.in_proj_weight
#         tensor = tensor.view(3, -1, tensor.shape[-1])
#         tensor = tensor.transpose(1, 2).contiguous()
#         return tensor

#     def ib(self) -> torch.Tensor:
#         tensor = self.in_proj_bias
#         tensor = tensor.view(3, 1, -1)
#         return tensor


class GroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        # self.attn = FlashAttention(in_channels, num_heads)
        self.attn = torch.nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

    def forward(self, x, pe):
        size = x.shape[0]
        
        batch_size = int(math.ceil(size / self.group_size))

        x = x.view(batch_size, self.group_size, -1)
        pe = pe.view(batch_size, self.group_size, -1)

        q = k = x + pe
        v = x
        x, _ = self.attn(q, k, v)

        # x = x.view(batch_size * self.group_size, -1)
        x = x.reshape(batch_size * self.group_size, -1)

        return x


class GroupGlobalAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        # self.attn = FlashAttention(in_channels, num_heads)
        self.attn = torch.nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

    def forward(self, x, pe):
        size = x.shape[0]
        batch_size = int(math.ceil(size / self.group_size))

        x = x.view(batch_size, self.group_size, -1)
        # pe = pe.view(batch_size, self.group_size, -1)

        tmp = x.permute((0, 2, 1))
        x = torch.nn.functional.adaptive_max_pool1d(tmp, output_size= 1).squeeze()
        
        q = k = x
        v = x
        x, _ = self.attn(q, k, v)

        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channels, num_heads, activation, group_size) -> None:
        super().__init__()
        self.attn = GroupAttention(in_channels, num_heads, group_size)

        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.act = _get_activation_fn(activation)

        self.fp16_enabled = False

    @auto_fp16(apply_to=("src", "pe"))
    def forward(self, src, pe):
        src = self.norm1(src + self.attn(src, pe))
        src = self.norm2(src + self.fc2(self.act(self.fc1(src))))              
        return src


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads,
        activation,
        group_size,
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList()
        for _ in range(4):
            layer = BasicLayer(
                in_channels,
                num_heads,
                activation,
                group_size=group_size,
            )
            self.block.append(layer)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str, Any]) -> torch.Tensor:
        for k, name in enumerate(["x", "x_shift", "y", "y_shift"]):
            indices = mappings[name]
            
            diff = len(mappings["flat2win"]) - len(mappings["win2flat"])
            
            x_flat2win = x[indices]            
            x_flat2win = torch.nn.functional.pad(x_flat2win, (0,0,0, diff))
            x_flat2win = x_flat2win[mappings["flat2win"]]

            pe_flat2win =pe[indices]
            pe_flat2win = torch.nn.functional.pad(pe_flat2win, (0,0,0, diff))
            pe_flat2win = pe_flat2win[mappings["flat2win"]]

            block = self.block[k]
            out = block(x_flat2win, pe_flat2win)
            x[indices] = out[mappings["win2flat"]]

        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu


@torch.inference_mode()
def get_window_coors_shift(coords, sparse_shape, window_shape, shifted):
    n, m, _ = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    if shifted:
        shift_x, shift_y = (n2 // 2, m2 // 2)
        x = coords[:, 3] + shift_x
        y = coords[:, 2] + shift_y
    else:
        x = coords[:, 3]
        y = coords[:, 2]

    x1 = x // n2 # window coordinate
    y1 = y // m2
    x2 = x % n2  # index within pixel cocord
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
        self,
        window_shape,
        sparse_shape,
        group_size,
    ) -> None:
        super().__init__()
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.group_size = group_size

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)
        for i in range(batch_size):
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                    - self.group_size
                ]
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}
        for shifted in [False, True]:
            (
                n2,
                m2,
                n1,
                m1,
                x1,
                y1,
                x2,
                y2,
            ) = get_window_coors_shift(coords, self.sparse_shape, self.window_shape, shifted=shifted)
            vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        return mappings


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        feat_dim,
        sparse_shape,
        normalize_pos,
        pos_temperature,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.sparse_shape = sparse_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.inv_freq_yoko = self.inv_freq()[None, :]

    def forward(self, coors, dtype):
        size_x, size_y, size_z = self.sparse_shape

        x, y = coors[:, 3], coors[:, 2]

        if self.normalize_pos:
            x = x / size_x * 2 * 3.1415  # [-pi, pi]
            y = y / size_y * 2 * 3.1415  # [-pi, pi]

        # inv_freq = self.inv_freq

        # [num_tokens, pos_length]
        # pex = x[:, None] / inv_freq()[None, :]
        # pey = y[:, None] / inv_freq()[None, :]
        pex = x[:, None] / self.inv_freq_yoko
        pey = y[:, None] / self.inv_freq_yoko


        # [num_tokens, pos_length]
        pex = torch.stack([pex[:, ::2].sin(), pex[:, 1::2].cos()], dim=-1).flatten(1)
        pey = torch.stack([pey[:, ::2].sin(), pey[:, 1::2].cos()], dim=-1).flatten(1)
        pe = torch.cat([pex, pey], dim=-1).to(dtype)

        gap = self.feat_dim - pe.size(1)
        if gap > 0:
            pe_p = torch.zeros((pe.size(0), gap), dtype=dtype, device=coors.device)
            pe = torch.cat([pe, pe_p], dim=1)

        return pe

    def inv_freq(self):
        ndim = 2
        pos_length = (self.feat_dim // (ndim * 2)) * 2

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device="cuda")
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        return inv_freq


# @MIDDLE_ENCODERS.register_module()
class FlatFormer(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_heads=8,
        num_blocks=2,
        activation="gelu",
        window_shape=(9, 9, 1),
        sparse_shape=(468, 468, 1),
        output_shape=(468, 468),
        pos_temperature=10000,
        normalize_pos=False,
        # group_size=69,
        group_size=64,
        # group_size=128,
        # group_size=256,
    ) -> None:
        super().__init__()
        self.group_size = group_size

        self.embedding = PositionalEmbedding(in_channels, sparse_shape, normalize_pos, pos_temperature)
        self.mapping = FlattenedWindowMapping(
            window_shape=window_shape,
            sparse_shape=sparse_shape,
            group_size=group_size,
        )

        sparse_shape = (sparse_shape[0]//4, sparse_shape[1]//4, 1)
        self.embedding_down = PositionalEmbedding(in_channels, sparse_shape, normalize_pos, pos_temperature)
        self.mapping_down = FlattenedWindowMapping(
            window_shape=window_shape,
            sparse_shape=sparse_shape,
            group_size=group_size,
        )

        self.block_list = nn.ModuleList()
        for _ in range(num_blocks):
            self.block_list.append(BasicBlock(in_channels, num_heads, activation, group_size))

        self._reset_parameters()

        self.output_shape = output_shape
        # self.output_shape = (output_shape[0]//4,output_shape[1]//4)

    @auto_fp16(apply_to=('x',))
    def forward(self, x, coords, batch_size):

        # x, coords = sparse_max_pool_1d(x, coords, kernel_size=4)

        torch.cuda.synchronize()
        start_time_1 = time.time()
        mappings = self.mapping(coords, batch_size)

        torch.cuda.synchronize()
        end_time_1 = time.time()
        start_time_2 = time.time()
        pe = self.embedding(coords, x.dtype)


        torch.cuda.synchronize()
        end_time_2 = time.time()
        start_time_3 = time.time()
        for _, block in enumerate(self.block_list):
            x = block(x, pe, mappings)

        torch.cuda.synchronize()
        end_time_3 = time.time()
        start_time_4 = time.time()

        # diff = len(mappings["flat2win"]) - len(mappings["win2flat"])
        # x_flat2win = x[mappings["x"]]
        # print(x_flat2win.shape)
        # x_flat2win = torch.nn.functional.pad(x_flat2win, (0,0,0, diff))
        # print(x_flat2win.shape)
        # x_flat2win = x_flat2win[mappings["flat2win"]]
        # print(x_flat2win.shape)
        # x_flat2win = x_flat2win.view(-1 ,64,x_flat2win.shape[1])
        # print(x_flat2win.shape)
        # x_flat2win = x_flat2win.permute((0, 2, 1))
        # print(x_flat2win.shape)
        # x_flat2win = torch.nn.functional.adaptive_max_pool1d(x_flat2win, output_size= 1).squeeze()
        # x_flat2win = x_flat2win.view(batch_size, -1, x_flat2win.shape[-1])
        # print(x_flat2win.shape)
        # x, coords = sparse_max_pool_1d(x, coords, kernel_size=4)
        # print(x.shape)

        torch.cuda.synchronize()
        end_time_4 = time.time()
        start_time_5 = time.time()
        # pe = self.embedding(coords, x.dtype)

        torch.cuda.synchronize()
        end_time_5 = time.time()
        start_time_6 = time.time()
        # mappings = self.mapping_down(coords, batch_size)


        torch.cuda.synchronize()
        end_time_6 = time.time()
        start_time_7 = time.time()

        # for _, block in enumerate(self.block_list):
        #     x = block(x, pe, mappings)

        torch.cuda.synchronize()
        end_time_7 = time.time()
        start_time_8 = time.time()

        # x = self.recover_bev(x, coords, batch_size)

        torch.cuda.synchronize()
        end_time_8 = time.time()
        
        elapse_1 =  (end_time_1 - start_time_1) * 1e3
        elapse_2 =  (end_time_2 - start_time_2) * 1e3
        elapse_3 =  (end_time_3 - start_time_3) * 1e3
        elapse_4 =  (end_time_4 - start_time_4) * 1e3
        elapse_5 =  (end_time_5 - start_time_5) * 1e3
        elapse_6 =  (end_time_6 - start_time_6) * 1e3
        elapse_7 =  (end_time_7 - start_time_7) * 1e3
        elapse_8 =  (end_time_8 - start_time_8) * 1e3
        print(f"flatformer= {elapse_1:.2f}, {elapse_2:.2f}, {elapse_3:.2f}, {elapse_4:.2f}, {elapse_5:.2f}, {elapse_6:.2f}, {elapse_7:.2f}, {elapse_8:.2f}")


        # return x

    def _reset_parameters(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(feat_dim, nx * ny, dtype=voxel_feat.dtype, device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]  # [n, c]
            voxels = voxels.t()  # [c, n]
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas


from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.tensor_cache import TensorCache
import time

def generate_random_point_cloud(size, voxel_size):
    pc = np.fromfile(
        "data/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
        dtype=np.float32,
    ).reshape(-1, 5)[:, :4]
    labels = np.random.choice(10, size)

    coords, feats = pc[:, :3], pc

    # coords[:, [2,1,0]] = coords


    coords -= np.min(coords, axis=0, keepdims=True)

    torch.cuda.synchronize()
    start_time = time.time()
    coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
    torch.cuda.synchronize()
    end_time = time.time()
    print("quantization=", (end_time - start_time) * 1000, "[ms]")


    # print(coords.shape, indices.shape)
    # coords = coords[:-1]
    # indices = indices[:-1]
    # print(coords.shape, indices.shape)

    torch.cuda.synchronize()
    start_time = time.time()
    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats[indices], dtype=torch.float)
    labels = torch.tensor(labels[indices], dtype=torch.long)
    input = SparseTensor(coords=coords, feats=feats)
    # label = SparseTensor(coords=coords, feats=labels)
    torch.cuda.synchronize()
    end_time = time.time()
    print("SparseTensor=", (end_time - start_time) * 1000, "[ms]")

    feed_dict = {"input": input}

    return feed_dict


def generate_batched_random_point_clouds(size, voxel_size, batch_size=2):
    batch = []
    for _ in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)

if __name__ == "__main__":
    NUM_PC_CHANNELS = 4
    VOXEL_SIZE = (0.1, 0.1, 0.1)
    MAX_X = 200
    MAX_Y = 200
    MAX_Z = 4
    NUM_PC = MAX_X * MAX_Y * MAX_Z
    BATCH_SIZE = 2

    feed_dict = generate_batched_random_point_clouds(size=NUM_PC,
                                                     voxel_size=VOXEL_SIZE,
                                                     batch_size=BATCH_SIZE)


    proj = torch.nn.Linear(4,128)
    feed_dict["input"].feats = proj(feed_dict["input"].feats)


    # input = 
    model = FlatFormer()
    model = model.cuda()
    inputs = feed_dict["input"].to("cuda")

    with torch.no_grad():
        for iter in range(20):
            torch.cuda.synchronize()
            start_time = time.time()
            output = model(inputs.feats, inputs.coords, batch_size=2)
            torch.cuda.synchronize()
            end_time = time.time()
            # print("Model=", (end_time - start_time) * 1000, "[ms]")