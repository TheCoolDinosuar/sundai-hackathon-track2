"""Offline weight quantization with a tiled scale layout.

Weights stay row-major packed so the benchmark still sees the correct `N`
dimension, but the per-group scales are pre-arranged in 128-column tiles to
make the runtime kernel's scale fetches cheaper.
"""

import torch


BLOCK_N = 128
BLOCK_K = 64


def quantize_weights(weight: torch.Tensor, group_size: int = 64) -> dict:
    """Quantize a FP16 weight tensor to packed INT4 format.

    Args:
        weight: [N, K] float16 weight tensor.
        group_size: Number of elements per quantization group.

    Returns:
        dict with:
            "weight_packed": [N, K//2] uint8 tensor (packed INT4)
            "weight_scales": either [N, K//group_size] or
                [N//128, K//group_size, 128] float16 tensor
            "group_size": int
    """
    assert weight.dim() == 2, "weight must be 2D [N, K]"
    N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 2 == 0, "group_size must be even"

    num_groups = K // group_size

    # Work in float32 for stable scale selection.
    w = weight.float().reshape(N, num_groups, group_size)

    # Mild percentile clipping preserves accuracy on outlier-heavy rows while
    # keeping the runtime format compatible with the symmetric INT4 kernel.
    abs_w = w.abs()
    clip = torch.quantile(abs_w, 0.992, dim=-1, keepdim=True)
    max_abs = abs_w.amax(dim=-1, keepdim=True)
    scale_base = torch.minimum(max_abs, clip * 1.05)
    scale = torch.where(scale_base > 0, scale_base / 7.0, torch.zeros_like(scale_base))
    rscale = torch.where(scale_base > 0, 7.0 / scale_base, torch.zeros_like(scale_base))

    q = (w * rscale).round().clamp(-8, 7).to(torch.int8)

    q = q.reshape(N, K)
    even = (q[:, 0::2] & 0xF).to(torch.uint8)
    odd = ((q[:, 1::2] & 0xF) << 4).to(torch.uint8)
    packed = (odd | even).contiguous()

    if group_size == BLOCK_K and N % BLOCK_N == 0:
        n_tiles = N // BLOCK_N
        scales = (
            scale.squeeze(-1)
            .reshape(n_tiles, BLOCK_N, num_groups)
            .permute(0, 2, 1)
            .contiguous()
            .half()
        )
    else:
        scales = scale.squeeze(-1).half()

    return {
        "weight_packed": packed,
        "weight_scales": scales,
        "group_size": group_size,
    }
