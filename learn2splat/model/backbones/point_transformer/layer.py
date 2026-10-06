import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import pointops
from einops import rearrange

from ..unimatch.dinov2.layers.block import Block as MultiViewBlock
from ..unimatch.utils import mv_feature_add_position

USE_PYTORCH_ATTN = False
USE_FLASH_ATTN3 = False

_simple_knn_indices = None  # bound lazily on first use of the simple_knn backend (see knn_search)


def knn_search(knn_samples, p, o, backend="simple_knn"):
    """Per-cloud K-NN neighbour indices [N, knn_samples] (self-inclusive, global indices).

    backend="simple_knn" (default): simple_knn.knn_indices (Morton/box-pruned ~O(N log N)).
    backend="pointops": pointops.knn_query (brute force ~O(N^2)) -- kept for comparison/A-B only.
    Both return identical neighbour sets.

    simple_knn is imported lazily here rather than at module load: the callers wrap the import
    of this module in `try/except: pass`, which would swallow a module-level raise -- importing
    at first use lets the mandatory build-instruction error actually reach the user.
    """
    if backend == "pointops":
        idx, _ = pointops.knn_query(knn_samples, p, o, p, o)
        return idx
    global _simple_knn_indices
    if _simple_knn_indices is None:
        try:
            from simple_knn._C import knn_indices
        except ImportError as e:
            raise ImportError(
                "simple_knn.knn_indices is required -- it is the point transformer's default KNN "
                "backend. Build the better-simple-knn submodule: `pip install submodules/simple-knn`."
            ) from e
        _simple_knn_indices = knn_indices
    idx, _ = _simple_knn_indices(p, knn_samples, o)  # the C++ side enforces contiguity
    return idx


class KNNAttention(nn.Module):
    def __init__(self, channels, knn_samples=16, no_rpe=True,
                 qk_norm=False,
                 num_heads=1,
                 proj_channels=None,
                 use_fused=False,
                 ):
        super().__init__()
        self.proj_channels = proj_channels

        self.knn_samples = knn_samples
        self.no_rpe = no_rpe
        self.num_heads = num_heads
        assert self.num_heads == 1

        self.use_fused = use_fused
        if use_fused:
            try:
                import sys
                from learn2splat.paths import PROJECT_DIR
                sys.path.append(str(PROJECT_DIR / "submodules"))
                from fused_knn_attn import fused_knn_attention, FUSED_KNN_ATTN_CUDA_AVAILABLE
                self._fused_knn_attention = fused_knn_attention
                if not FUSED_KNN_ATTN_CUDA_AVAILABLE:
                    import warnings
                    warnings.warn(
                        "Fused KNN attention CUDA extension not available, "
                        "using PyTorch fallback (still avoids [N,K,C] intermediates)"
                    )
            except ImportError:
                import warnings
                warnings.warn(
                    "fused_knn_attn package not found, falling back to unfused attention"
                )
                self.use_fused = False

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(channels)
            self.k_norm = nn.RMSNorm(channels)

        if self.proj_channels is not None:
            self.qkv = nn.Linear(channels, self.proj_channels * 3, bias=False)
            self.proj = nn.Linear(self.proj_channels, channels)
        else:
            self.qkv = nn.Linear(channels, channels * 3, bias=False)
            self.proj = nn.Linear(channels, channels)

        if not self.no_rpe:
            self.rpe = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )

    def forward(self, pxo, knn_idx=None):
        # [N, 3], [N, C], [B]
        p, x, o = pxo
        c = x.size(1)

        if self.proj_channels is not None:
            c = self.proj_channels

        assert c % self.num_heads == 0
        head_dim = c // self.num_heads
        scale_factor = head_dim ** -0.5

        qkv = self.qkv(x)  # [N, 3*C]
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)  # each [N, C]

        # ---- Fused path: gather + attention in one kernel ----
        if self.use_fused and self.no_rpe:
            # Ensure we have KNN indices
            if knn_idx is None:
                knn_idx, _ = pointops.knn_query(
                    self.knn_samples, p, o, p, o
                )

            # qk_norm: RMSNorm normalizes each C-dim vector independently,
            # so applying before gather is equivalent to applying after gather.
            if self.qk_norm:
                x_q = self.q_norm(x_q)
                x_k = self.k_norm(x_k)

            out = self._fused_knn_attention(
                x_q.contiguous(), x_k.contiguous(), x_v.contiguous(),
                knn_idx.contiguous(), scale_factor
            )
            out = self.proj(out)
            return out

        # ---- Unfused path: k and v concatenated so one gather serves both ----
        x_kv = torch.cat([x_k, x_v], dim=-1)  # [N, 2C/3]
        x_kv_query, _ = pointops.knn_query_and_group(
            x_kv.contiguous(), p, o, new_xyz=p, new_offset=o,
            idx=knn_idx, nsample=self.knn_samples, with_xyz=False
        )  # [N, K, 2C/3]
        x_k, x_v = torch.chunk(x_kv_query, chunks=2, dim=-1)

        # [N, K, 3], [N, K, C]
        # NOTE: without xyz in knn
        # p_r, x_k = x_k[:, :, :3], x_k[:, :, 3:]

        # [N, 1, K]
        assert self.no_rpe
        if not self.no_rpe:
            rpe = self.rpe(p_r).permute(0, 2, 1)
        else:
            rpe = 0

        if self.qk_norm:
            x_q = self.q_norm(x_q)
            x_k = self.k_norm(x_k)

        n, k, c = x_k.shape

        # attention
        if USE_PYTORCH_ATTN:
            out = F.scaled_dot_product_attention(
                x_q.view(n, 1, c),
                x_k.view(n, k, c),
                x_v.view(n, k, c),
            ).reshape(n, c)  # [N, C]

        elif (USE_FLASH_ATTN3 and FA3_AVAILABLE and self.no_rpe):
            # no relative pos enc
            out = flash_attn_func(
                x_q.view(n, 1, self.num_heads, head_dim).to(torch.bfloat16),
                x_k.view(n, k, self.num_heads, head_dim).to(torch.bfloat16),
                x_v.view(n, k, self.num_heads, head_dim).to(torch.bfloat16),
            )[0].reshape(n, c).float()  # [N, C]
        else:
            # [N, 1, K]
            scores = torch.matmul(x_q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale_factor + rpe
            # [N, C]
            out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)

        out = self.proj(out)

        return out


class MLP(nn.Module):
    def __init__(
            self,
            channels,
            act="gelu",
    ):
        super().__init__()

        expansion = 4

        self.fc1 = nn.Linear(channels, channels * expansion)
        if act is None or act in ['none', 'identity']:
            self.act = nn.Identity()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"unsupported activation {act}")
        self.fc2 = nn.Linear(channels * expansion, channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, knn_samples=16, post_norm=False,
                 no_rpe=False,
                 no_attn=False,
                 no_norm=False,
                 act="gelu",
                 qk_norm=False,
                 norm_pt_block=False,
                 num_heads=1,
                 attn_proj_channels=None,
                 use_fused_attn=False,
                 ):
        super().__init__()
        self.post_norm = post_norm
        self.no_attn = no_attn
        self.norm_pt_block = norm_pt_block

        if no_norm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

        if self.no_attn:
            self.linear = nn.Linear(channels, channels)
        else:
            self.attn = KNNAttention(channels, knn_samples=knn_samples, no_rpe=no_rpe,
                                     qk_norm=qk_norm,
                                     num_heads=num_heads,
                                     proj_channels=attn_proj_channels,
                                     use_fused=use_fused_attn,
                                     )
        self.mlp = MLP(channels, act=act)

        if self.norm_pt_block:
            self.norm3 = nn.LayerNorm(channels)

    def forward(self, pxo, knn_idx=None):
        p, x, o = pxo

        if self.post_norm:
            if self.no_attn:
                x = x + self.norm1(self.linear(x))
            else:
                x = x + self.norm1(self.attn((p, x, o), knn_idx=knn_idx))
            x = x + self.norm2(self.mlp(x))
        else:
            if self.no_attn:
                x = x + self.linear(self.norm1(x))
            else:
                x = x + self.attn((p, self.norm1(x), o), knn_idx=knn_idx)
            x = x + self.mlp(self.norm2(x))

        if self.norm_pt_block:
            x = self.norm3(x)

        return x


class PlainPointTransformer(nn.Module):
    def __init__(self, channels, knn_samples=16, num_blocks=4, post_norm=False,
                 no_rpe=False,
                 no_attn=False,
                 no_norm=False,
                 act="gelu",
                 qk_norm=False,
                 norm_pt_block=False,
                 num_heads=1,
                 attn_proj_channels=None,
                 cache_knn_idx=None,
                 knn_idx_update_every=1,
                 with_mv_attn=False,
                 with_mv_attn_lowres=False,
                 mv_attn_first=False,
                 no_mv_attn=False,
                 conv_with_norm=False,
                 mv_shuffle_attn=False,
                 with_pos_enc=False,
                 shuffle_attn_no_norm=False,
                 mv_unimatch_attn=False,
                 use_checkpointing=False,
                 init_use_checkpointing=False,
                 use_fused_attn=False,
                 knn_backend="simple_knn",
                 knn_pos_clamp=None,
                 ):
        super().__init__()

        self.cache_knn_idx = cache_knn_idx
        self.knn_idx_update_every = knn_idx_update_every
        self.knn_samples = knn_samples
        self.knn_backend = knn_backend
        # If set, the neighbour search below runs on a bounded COPY of the positions (see forward).
        self.knn_pos_clamp = knn_pos_clamp
        self.use_checkpointing = use_checkpointing
        self.init_use_checkpointing = init_use_checkpointing

        self.with_mv_attn = with_mv_attn
        self.with_mv_attn_lowres = with_mv_attn_lowres
        if with_pos_enc:
            assert mv_shuffle_attn

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerBlock(channels, knn_samples=knn_samples,
                                                post_norm=post_norm,
                                                no_rpe=no_rpe,
                                                no_attn=no_attn,
                                                no_norm=no_norm,
                                                act=act,
                                                qk_norm=qk_norm,
                                                norm_pt_block=norm_pt_block,
                                                num_heads=num_heads,
                                                attn_proj_channels=attn_proj_channels,
                                                use_fused_attn=use_fused_attn,
                                                ))

        # multi-view attention
        if self.with_mv_attn:
            self.mv_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                # if mv_shuffle_attn:
                if self.with_mv_attn_lowres:
                    self.mv_blocks.append(
                        MultViewLowresAttn(
                            channels,
                        )
                    )
                else:
                    self.mv_blocks.append(
                        MultiViewBlock(
                            channels,
                            num_heads=4,
                        )
                    )

    def forward(self, pxo, iter=0, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        # compute knn idx here only once and pass it to the model
        # the positions are not changed inside the blocks
        if self.cache_knn_idx is None or (iter % self.knn_idx_update_every) == 0:
            # The neighbour search's spatial indexing overflows on large-but-finite coordinates, so
            # knn_pos_clamp (opt-in) bounds a copy of the positions for the search only. The real p
            # reaches the attention blocks unchanged, and neighbour sets are unaffected because a
            # diverged Gaussian is far away either way.
            p_knn = p
            if self.knn_pos_clamp is not None:
                c = self.knn_pos_clamp
                p_knn = p_knn.clamp(min=-c, max=c)
            knn_idx = knn_search(self.knn_samples, p_knn, o, self.knn_backend)
            self.cache_knn_idx = knn_idx
        else:
            knn_idx = self.cache_knn_idx

        if self.with_mv_attn:
            assert b is not None and v is not None and h is not None and w is not None
            if self.use_checkpointing:
                raise NotImplementedError

            for i in range(len(self.blocks)):
                # knn attention
                x = self.blocks[i]([p, x, o], knn_idx=knn_idx)
                # global multi-view attention
                x = rearrange(x, "(b v h w) c -> b (v h w) c", b=b, v=v, h=h, w=w)
                if self.with_mv_attn_lowres:
                    x = self.mv_blocks[i](x, v=v, h=h, w=w)
                else:
                    x = self.mv_blocks[i](x)
                x = rearrange(x, "b (v h w) c -> (b v h w) c",
                              b=b, v=v, h=h, w=w)
        else:
            for blk in self.blocks:
                if self.init_use_checkpointing:
                    # checkpointing the inital reconstruction model
                    # NOTE: cannot cache knn_idx here, otherwise index out error
                    def custom_forward(p, x, o):
                        return blk((p, x, o), knn_idx=None)  # knn_idx is closed over

                    x = torch.utils.checkpoint.checkpoint(custom_forward, p, x, o)
                else:
                    x = blk((p, x, o), knn_idx=knn_idx)

        return x


class MultViewLowresAttn(nn.Module):
    def __init__(self, channels, no_mv_attn=False, with_pos_enc=False, shuffle_attn_no_norm=False,
                 down_factor=4,
                 attn_proj_channels=None,
                 ):
        super().__init__()

        self.down_factor = down_factor
        self.with_pos_enc = with_pos_enc

        self.attn_proj_channels = attn_proj_channels

        if attn_proj_channels:
            ori_channels = channels
            self.proj0 = nn.Linear(channels, attn_proj_channels)
            channels = attn_proj_channels

        if self.down_factor == 8:
            down_factor = 4
        else:
            down_factor = self.down_factor

        self.proj1 = nn.Linear(channels * down_factor ** 2, channels)
        if shuffle_attn_no_norm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.LayerNorm(channels)

        self.proj2 = nn.Linear(channels, channels * down_factor ** 2)

        if shuffle_attn_no_norm:
            self.norm2 = nn.Identity()
        else:
            self.norm2 = nn.LayerNorm(channels * down_factor ** 2)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if attn_proj_channels:
            self.proj3 = nn.Linear(channels, ori_channels)

        if no_mv_attn:
            self.attn = nn.Identity()
        else:
            num_heads = 1 if self.attn_proj_channels else 4
            self.attn = MultiViewBlock(channels, num_heads, no_attn=no_mv_attn)

    def forward(self, x, v=None, h=None, w=None, y=None):
        if y is not None:
            return self.forward_cross_attn(x, y, v, h, w)
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)

        if self.with_pos_enc:
            x = mv_feature_add_position(x, attn_splits=1, feature_channels=x.size(1))
            # print(x.shape)

        if self.down_factor == 8:
            # bilinear to half first to save channels
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = self.proj1(x)
        x = self.norm1(x)

        x = self.attn(x)

        x = self.proj2(x)
        x = self.norm2(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down_factor, w=w // self.down_factor)
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            # bilinear to full
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x

    def forward_cross_attn(self, x, y, v=None, h=None, w=None):
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        assert y is not None
        y = rearrange(y, "b (v h w) c -> (b v) c h w", h=h, w=w)  # different v with x
        num_cross_view = y.shape[0] // x.shape[0]

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)

        if self.with_pos_enc:
            x = mv_feature_add_position(x, attn_splits=1, feature_channels=x.size(1))
            # print(x.shape)

        if self.down_factor == 8:
            # bilinear to half first to save channels
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)
        y = F.pixel_unshuffle(y, down_factor)

        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        y = rearrange(y, "(b v) c h w -> b (v h w) c", v=num_cross_view)
        x = self.proj1(x)
        x = self.norm1(x)

        y = self.proj1(y)
        y = self.norm1(y)

        # x_tmp = self.attn(x)

        # print((x - y).abs().max().item())

        x = self.attn(x, y)

        # there will be slight diff for self and cross attn caused by flash3
        # print((x_tmp - x).abs().max().item())

        x = self.proj2(x)
        x = self.norm2(x)

        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down_factor, w=w // self.down_factor)
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            # bilinear to full
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x


class PointLinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, pxo, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        x = self.linear(x)

        return [p, x, o]

