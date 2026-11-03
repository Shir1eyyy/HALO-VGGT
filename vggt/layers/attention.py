import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm.std import tqdm

from merging.merge import token_merge_bipartite2d


XFORMERS_AVAILABLE = False

# Global state used only by the original attention-map visualization utility.
vis_attn_map = False
current_images = []
attention_map = None


class Attention(nn.Module):
    """VGGT attention with optional FastVGGT or random K/V reduction.

    ``method="random"`` keeps all query tokens and reduces only keys and values in
    global-attention blocks. The first view is kept intact; tokens from later views
    are selected independently per view. ``rand_token_select`` controls whether
    those tokens are sampled pseudo-randomly or at uniform intervals.
    """

    SUPPORTED_METHODS = {"vggt", "fastvggt", "random"}
    SUPPORTED_TOKEN_SELECTORS = {"random", "uniform"}

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        kv_group_size: int = 1,
        fused_attn: bool = True,
        rope=None,
        global_merging=None,
        patch_width: int = 37,
        patch_height: int = 28,
        merge_ratio: float = 0.9,
        method: str = "fastvggt",
        keep_ratio: float = 0.05,
        level1: float = 0.1,
        level2: float = 0.2,
        level3: float = 0.3,
        level4: float = 0.4,
        head_group_count: int = 4,
        causal_online_controller: bool = True,
        causal_controller_enter_threshold: float = 0.7,
        causal_controller_head_topk: int = 4,
        rand_token_select: str = "random",
        random_seed: int = 33,
        random_keep_first_frame: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method={method!r}; choose from {sorted(self.SUPPORTED_METHODS)}"
            )
        if rand_token_select not in self.SUPPORTED_TOKEN_SELECTORS:
            raise ValueError(
                "rand_token_select must be either 'random' or 'uniform', "
                f"got {rand_token_select!r}"
            )
        if head_group_count not in (2, 4):
            raise ValueError("head_group_count must be 2 or 4")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.fused_attn = fused_attn
        self.merge_ratio = merge_ratio
        self.method = method
        self.keep_ratio = self._validate_ratio("keep_ratio", keep_ratio)
        self.levels = tuple(
            self._validate_ratio(name, value)
            for name, value in (
                ("level1", level1),
                ("level2", level2),
                ("level3", level3),
                ("level4", level4),
            )
        )
        self.head_group_count = head_group_count
        self.causal_online_controller = causal_online_controller
        self.causal_controller_enter_threshold = causal_controller_enter_threshold
        self.causal_controller_head_topk = causal_controller_head_topk
        self.rand_token_select = rand_token_select
        self.random_seed = int(random_seed)
        self.random_keep_first_frame = random_keep_first_frame

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.kv_group_size = kv_group_size

    @staticmethod
    def _validate_ratio(name: str, value: float) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
        return value

    @staticmethod
    def _uniform_indices(size: int, count: int, device: torch.device) -> Tensor:
        if count <= 0 or size <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        if count >= size:
            return torch.arange(size, device=device, dtype=torch.long)
        if count == 1:
            return torch.tensor([size // 2], device=device, dtype=torch.long)

        indices = (
            torch.linspace(0, size - 1, steps=count, device=device)
            .round()
            .long()
            .unique(sorted=True)
        )
        if indices.numel() < count:
            mask = torch.ones(size, device=device, dtype=torch.bool)
            mask[indices] = False
            remaining = torch.arange(size, device=device)[mask]
            indices = torch.cat([indices, remaining[: count - indices.numel()]])
            indices = indices.sort().values
        return indices

    def _make_generator(
        self, device: torch.device, block_id: int, group_id: int
    ) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(
            self.random_seed * 10_000_000 + block_id * 100 + group_id
        )
        return generator

    def _sample_indices(
        self,
        size: int,
        count: int,
        device: torch.device,
        batch_shape: tuple[int, ...],
        block_id: int,
        group_id: int,
    ) -> Tensor:
        count = max(0, min(int(count), int(size)))
        if self.rand_token_select == "uniform":
            indices = self._uniform_indices(size, count, device)
            view_shape = (1,) * len(batch_shape) + (count,)
            return indices.view(*view_shape).expand(*batch_shape, count)

        scores = torch.rand(
            *batch_shape,
            size,
            device=device,
            generator=self._make_generator(device, block_id, group_id),
        )
        indices = scores.topk(count, dim=-1, sorted=False).indices
        return indices.sort(dim=-1).values

    def _sample_per_view(
        self,
        num_keys: int,
        keep_ratio: float,
        device: torch.device,
        batch_shape: tuple[int, ...],
        block_id: int,
        group_id: int,
    ) -> Tensor:
        """Select token indices, preserving view boundaries and the first view."""
        per_view = self.patch_width * self.patch_height + 5
        if per_view > 0 and num_keys > per_view and num_keys % per_view == 0:
            num_views = num_keys // per_view
            parts = []
            first_sampled_view = 0
            if self.random_keep_first_frame:
                first = torch.arange(per_view, device=device, dtype=torch.long)
                view_shape = (1,) * len(batch_shape) + (per_view,)
                parts.append(first.view(*view_shape).expand(*batch_shape, per_view))
                first_sampled_view = 1

            remaining_views = num_views - first_sampled_view
            if remaining_views > 0:
                keep_per_view = max(
                    1, min(per_view, int(round(per_view * keep_ratio)))
                )
                relative = self._sample_indices(
                    size=per_view,
                    count=keep_per_view,
                    device=device,
                    batch_shape=(*batch_shape, remaining_views),
                    block_id=block_id,
                    group_id=group_id,
                )
                bases = (
                    torch.arange(
                        first_sampled_view, num_views, device=device, dtype=torch.long
                    )
                    * per_view
                )
                bases = bases.view(*((1,) * len(batch_shape)), remaining_views, 1)
                parts.append((relative + bases).flatten(-2))

            return torch.cat(parts, dim=-1).sort(dim=-1).values

        keep_count = max(1, min(num_keys, int(round(num_keys * keep_ratio))))
        return self._sample_indices(
            size=num_keys,
            count=keep_count,
            device=device,
            batch_shape=batch_shape,
            block_id=block_id,
            group_id=group_id,
        )

    def _sampled_qk_margin(self, q: Tensor, k: Tensor) -> Tensor:
        """Estimate each head's top-1/top-2 attention-score margin cheaply."""
        with torch.no_grad():
            query_indices = self._uniform_indices(q.shape[2], 32, q.device)
            key_indices = self._uniform_indices(k.shape[2], 4096, k.device)
            sampled_q = q.index_select(2, query_indices).float()
            sampled_k = k.index_select(2, key_indices).float()
            scores = sampled_q @ sampled_k.transpose(-2, -1)
            scores = scores * self.scale
            if scores.shape[-1] < 2:
                return torch.zeros(q.shape[1], device=q.device, dtype=torch.float32)
            
            probs = torch.softmax(scores, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            
            # top2 = scores.topk(2, dim=-1).values
            return (top2[..., 0] - top2[..., 1]).mean(dim=(0, 2))

    def _should_use_head_adaptive(self, q: Tensor, k: Tensor) -> tuple[bool, Tensor | None]:
        # With the controller disabled, every reduced block uses per-head levels.
        if not self.causal_online_controller:
            return True, None
        margins = self._sampled_qk_margin(q, k)
        topk = max(1, min(self.causal_controller_head_topk, margins.numel()))
        score = margins.topk(topk).values.mean()
        return bool(score.item() > self.causal_controller_enter_threshold), margins

    def _head_plan(self, margins: Tensor) -> tuple[Tensor, Tensor]:
        group_levels = (
            (self.levels[0], self.levels[3])
            if self.head_group_count == 2
            else self.levels
        )
        sorted_heads = margins.argsort(descending=False)
        keep_ratios = torch.empty_like(margins, dtype=torch.float32)
        head_groups = torch.empty_like(sorted_heads)

        base, remainder = divmod(self.num_heads, self.head_group_count)
        start = 0
        for group_id, ratio in enumerate(group_levels, start=1):
            size = base + (1 if group_id <= remainder else 0)
            head_ids = sorted_heads[start : start + size]
            keep_ratios[head_ids] = ratio
            head_groups[head_ids] = group_id
            start += size
        return keep_ratios, head_groups

    @staticmethod
    def _gather_kv(k: Tensor, v: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
        gather_indices = indices[..., None].expand(*indices.shape, k.shape[-1])
        return (
            torch.gather(k, dim=2, index=gather_indices),
            torch.gather(v, dim=2, index=gather_indices),
        )

    def _head_adaptive_attention(
        self, q: Tensor, k: Tensor, v: Tensor, margins: Tensor, block_id: int
    ) -> Tensor:
        keep_ratios, head_groups = self._head_plan(margins)
        # SDPA returns the value tensor's dtype. With q/k normalization under
        # mixed precision, q can be float32 while v remains bfloat16, so the
        # destination must follow v rather than q for index_copy_.
        output = torch.empty(
            q.shape,
            device=q.device,
            dtype=v.dtype,
        )
        batch_size = q.shape[0]

        for group_id in range(1, self.head_group_count + 1):
            head_ids = torch.nonzero(head_groups == group_id, as_tuple=False).flatten()
            if head_ids.numel() == 0:
                continue
            ratio = float(keep_ratios[head_ids[0]].item())
            indices = self._sample_per_view(
                num_keys=k.shape[2],
                keep_ratio=ratio,
                device=k.device,
                batch_shape=(batch_size, int(head_ids.numel())),
                block_id=block_id,
                group_id=group_id,
            )
            q_group = q.index_select(1, head_ids)
            k_group = k.index_select(1, head_ids)
            v_group = v.index_select(1, head_ids)
            k_group, v_group = self._gather_kv(k_group, v_group, indices)
            group_output = F.scaled_dot_product_attention(
                q_group,
                k_group,
                v_group,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            if group_output.dtype != output.dtype:
                group_output = group_output.to(output.dtype)
            output.index_copy_(1, head_ids, group_output)
        return output

    def _random_kv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, block_id: int
    ) -> Tensor:
        use_head_adaptive, margins = self._should_use_head_adaptive(q, k)
        if use_head_adaptive:
            if margins is None:
                margins = self._sampled_qk_margin(q, k)
            return self._head_adaptive_attention(q, k, v, margins, block_id)

        indices = self._sample_per_view(
            num_keys=k.shape[2],
            keep_ratio=self.keep_ratio,
            device=k.device,
            batch_shape=(k.shape[0],),
            block_id=block_id,
            group_id=0,
        )
        gather_indices = indices[:, None, :, None].expand(
            k.shape[0], k.shape[1], indices.shape[-1], k.shape[-1]
        )
        k = torch.gather(k, dim=2, index=gather_indices)
        v = torch.gather(v, dim=2, index=gather_indices)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

    def _visualize_attention(self, q: Tensor, k: Tensor) -> None:
        if not vis_attn_map:
            return
        chunk_size = 4096
        attn_maps = []
        total_chunks = (q.size(-2) + chunk_size - 1) // chunk_size
        start_time_total = time.time()
        with torch.no_grad():
            for start in tqdm(
                range(0, q.size(-2), chunk_size),
                total=total_chunks,
                desc="Processing chunks",
            ):
                end = min(start + chunk_size, q.size(-2))
                attn_maps.append((q[:, :, start:end] @ k.transpose(-2, -1)).cpu())
        print(f"Attention maps computed in {time.time() - start_time_total:.4f} seconds")

        attn = torch.cat(attn_maps, dim=-2)[0].mean(0)
        frame_token_num = self.patch_height * self.patch_width + 5
        target_tokens = [
            0,
            self.patch_height * self.patch_width,
            self.patch_height * self.patch_width * 10,
        ]
        for target_token_idx in target_tokens:
            if target_token_idx >= attn.shape[0]:
                continue
            for image_idx, image_path in enumerate(current_images):
                target_attn = attn[
                    target_token_idx,
                    image_idx * frame_token_num : (image_idx + 1) * frame_token_num,
                ]
                if target_attn.numel() != frame_token_num:
                    continue
                target_attn = target_attn[5:].reshape(
                    self.patch_height, self.patch_width
                )
                image = Image.open(image_path).convert("RGB")
                img_width, img_height = image.size
                target_attn = F.interpolate(
                    target_attn[None, None],
                    size=(img_height, img_width),
                    mode="bilinear",
                ).squeeze()
                target_attn = (target_attn - target_attn.min()) / (
                    target_attn.max() - target_attn.min() + 1e-12
                )
                color = plt.get_cmap("jet")(target_attn.float().numpy())[:, :, :3]
                overlay = np.asarray(image) / 255.0 * 0.5 + color * 0.5
                scene_name = Path(image_path).parts[-4]
                output_dir = (
                    f"attention_map/{scene_name}/block_{attention_map}/"
                    f"token_{target_token_idx}"
                )
                os.makedirs(output_dir, exist_ok=True)
                plt.imshow(overlay)
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, f"color_{image_idx}.png"))
                plt.close()

    def forward(
        self,
        x: Tensor,
        pos=None,
        global_merging=None,
        attn_type=None,
        block_id=None,
    ) -> Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        reduction_active = global_merging is not None
        if reduction_active and vis_attn_map:
            self._visualize_attention(q, k)

        unmerge = None
        if reduction_active and self.method == "fastvggt":
            generator = torch.Generator(device=x.device)
            generator.manual_seed(self.random_seed)
            merge_count = int(x.shape[1] * self.merge_ratio)
            merge, unmerge = token_merge_bipartite2d(
                x,
                self.patch_width,
                self.patch_height,
                2,
                2,
                merge_count,
                False,
                generator,
                enable_protection=True,
            )
            q_in = q.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
            k_in = k.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
            v_in = v.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, channels)
            q_out, k_out, v_out = merge(
                q_in, mode="mean", extra_tensors=k_in, extra_tensors_2=v_in
            )
            num_tokens = q_out.shape[1]
            q = q_out.reshape(
                batch_size, num_tokens, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
            k = k_out.reshape(
                batch_size, num_tokens, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
            v = v_out.reshape(
                batch_size, num_tokens, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        if reduction_active and self.method == "random":
            output = self._random_kv_attention(
                q, k, v, int(block_id if block_id is not None else global_merging)
            )
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        output = output.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        output = self.proj_drop(self.proj(output))
        if unmerge is not None:
            output = unmerge(output)
        return output


class MemEffAttention(Attention):
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        pos=None,
        global_merging=None,
        attn_type=None,
        block_id=None,
    ) -> Tensor:
        if pos is not None and self.rope is None:
            raise AssertionError("Position encoding is only supported with RoPE")
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(
                x,
                pos=pos,
                global_merging=global_merging,
                attn_type=attn_type,
                block_id=block_id,
            )

        # Kept for API compatibility if xFormers support is restored later.
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.unbind(2)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        output = torch.matmul(F.softmax(attn, dim=-1), v).reshape(
            batch_size, num_tokens, channels
        )
        return self.proj_drop(self.proj(output))
