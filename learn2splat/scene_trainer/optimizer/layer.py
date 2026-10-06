import functools

import torch
from jaxtyping import Float
from torch import nn as nn, Tensor


def _lazy_compile(**compile_kwargs):
    """Like ``@torch.compile`` but defers the (heavy) dynamo/inductor import
    until the function is first called. Applying ``torch.compile`` at import
    eagerly loads the compile stack (~3.5s); this keeps it off the import path
    and skips it entirely for runs that never call the wrapped function."""
    def decorator(fn):
        compiled = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                compiled = torch.compile(fn, **compile_kwargs)
            return compiled(*args, **kwargs)

        return wrapper

    return decorator


class SlicedG3RNorm(nn.Module):
    def __init__(self, num_features, input_slice, eps=1e-8):
        """
        Apply G3R normalization to a slice of the input features.

        Divide the input by the maximum absolute value in each channel.

        Args:
            num_features (int): Total number of features (channels).
            input_slice (slice): Channel slice to normalize independently.
            eps (float): Small constant to prevent division by zero.
        """
        super().__init__()
        self.input_slice = input_slice
        dummy = torch.zeros(1, num_features)
        chunk = dummy[:, input_slice]

        self.slice_size = chunk.shape[-1]
        self.eps = eps

    def forward(self, x: Float[Tensor, "B C"]):
        """
        Args:
            x (Tensor): Shape (B, C) where C = num_features
        Returns:
            Tensor: Same shape, only subset of channels normalized
        """
        # Split input into the slice to normalize and the rest
        chunk = x[:, self.input_slice]

        # Compute max absolute value per channel
        # Detach to avoid backpropagating through max operation
        max_val_per_channel = chunk.abs().max(0, keepdim=True)[0].detach() + self.eps

        # Apply G3R normalization to the selected slice
        # Replace the normalized slice back into the original input
        x = x.clone()
        x[:, self.input_slice] = chunk / max_val_per_channel
        return x


class CustomGroupNorm(nn.Module):
    def __init__(self, group_sizes, eps=1e-8, affine=True):
        """
        Args:
            group_sizes (list[int]): List of channel counts for each group. Must sum to total input channels.
            eps (float): Small constant to prevent division by zero.
            affine (bool): Whether to include learnable scale and bias per group.
        """
        super().__init__()
        self.group_sizes = group_sizes
        self.total_channels = sum(group_sizes)
        self.eps = eps

        # Create a LayerNorm module for each group
        self.group_norms = nn.ModuleList([
            nn.LayerNorm([size], eps=eps, elementwise_affine=affine)
            for size in group_sizes
        ])

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (B, C)
        Returns:
            Tensor: Same shape, group-wise normalized
        """
        B, C = x.shape
        assert C == self.total_channels, (
            f"Input has {C} channels, expected {self.total_channels} from group sizes {self.group_sizes}"
        )

        # Split input into channel groups
        splits = torch.split(x, self.group_sizes, dim=1)
        normed = []
        for i, g in enumerate(splits):
            normed_group = self.group_norms[i](g)
            normed.append(normed_group)

        return torch.cat(normed, dim=1)


class AdamState:
    def __init__(self, m, v, t):
        self.m = m  # First moment vector
        self.v = v  # Second moment vector
        self.t = t  # Time step


def slice_length(s, dim):
    step = s.step or 1
    start = s.start if s.start is not None else (0 if step > 0 else dim - 1)
    stop = s.stop if s.stop is not None else (dim if step > 0 else -1)
    if start < 0: start += dim
    if stop < 0: stop += dim
    start = max(0, min(dim, start))
    stop = max(0, min(dim, stop))
    return max(0, (stop - start + (step - 1)) // step) if step > 0 else \
           max(0, (start - stop + (-step - 1)) // -step)


@_lazy_compile(dynamic=True)
def _adam_smooth_unmasked(m, v, t, chunk, beta1, beta2, eps) -> Tensor:
    """Fused moment update + bias-corrected output for the unmasked path."""
    m.lerp_(chunk, 1 - beta1)
    v.mul_(beta2).addcmul_(chunk, chunk, value=1 - beta2)
    t_bc = t.reshape(t.shape[0], *([1] * (m.ndim - 1)))
    bias1 = 1 - beta1 ** t_bc
    bias2_sqrt = (1 - beta2 ** t_bc).sqrt_()
    denom = v.sqrt().div_(bias2_sqrt).add_(eps)
    return m.div(bias1).div_(denom)


@_lazy_compile(dynamic=True)
def _adam_smooth_masked(m, v, t, sel, chunk, beta1, beta2, eps) -> Tensor:
    """Fused moment update + bias-corrected output for the masked path."""
    m_sel = m[sel].lerp_(chunk, 1 - beta1)
    m[sel] = m_sel

    v_sel = v[sel].mul_(beta2).addcmul_(chunk, chunk, value=1 - beta2)
    v[sel] = v_sel

    t[sel] += 1

    t_sel = t[sel].reshape(-1, *([1] * (m.ndim - 1)))
    m_hat = m_sel / (1 - beta1 ** t_sel)
    v_hat = v_sel / (1 - beta2 ** t_sel)
    return m_hat / (torch.sqrt(v_hat) + eps)


class AdamInputSmoothing(nn.Module):
    """Per-Gaussian Adam-like smoother for the optimizer's input gradients: ``m`` (exp_avg), ``v``
    (exp_avg_sq) and a per-Gaussian step ``t``. Used both for the AdamOptimizer baseline's
    per-parameter smoothers and the learned optimizer's input normalizer.

    ADC behavior — these Adam moments follow FastGS and are NOT governed by the per-op latent-state
    policy (``clone/split/relocate/add_gaussian_state``; that concerns the learned *latent* state and
    is applied by ``KnnBasedOptimizerState``). The ADC helpers (``adc/base.py``) call
    ``clone``/``split``/``replace`` with no ``zero_t``, so the FastGS-reference default
    (``zero_t=False``) holds:
      - clone/split: new Gaussians get zeroed moments (m, v) and inherit the parent's step ``t`` — the
        per-Gaussian analog of FastGS zeroing ``exp_avg``/``exp_avg_sq`` while its global step carries on.
      - relocate (``replace``): the relocated Gaussian inherits the sampled-alive source's moments and step.
      - add / opacity-reset (``zero_out``): fresh zeroed moments (step kept on opacity reset).
    The ``zero_t=True`` branch is a primitive kept for the unit tests; ADC never sets it.
    """

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-15, input_slice: slice | None = None,
                 shape: tuple | None = None,
                 device=None):
        """
        Implements Adam-like smoothing for input vectors.

        Args:
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            eps (float): Small constant to prevent division by zero.
            input_slice (slice, optional): If provided, only apply smoothing to this slice of the input.
        """
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.input_slice: slice | None = input_slice
        if self.input_slice is not None:
            assert isinstance(self.input_slice, slice), "input_slice must be a slice or None"

        # Initialize first and second moment vectors
        if shape is None:
            self.reset()
        else:
            self.initialize(shape,
                            device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Adam-like smoothing to the input.

        Args:
            x (Tensor): Input tensor of shape (..., input_dim)

        Returns:
            Tensor: Smoothed tensor of same shape as input
        """
        # Select the relevant slice of the input
        chunk = x[..., self.input_slice] if self.input_slice is not None else x

        # Initialize internal state if needed
        if self.is_reset():
            self.initialize(chunk.shape, device=chunk.device)

        chunk_detached = chunk.detach()

        if self.sel is None:
            # Increment step first (matches PyTorch Adam convention)
            self.t += 1
            # Fused moment update + bias-corrected output (compiled kernel)
            chunk_smoothed = _adam_smooth_unmasked(self.m, self.v, self.t, chunk_detached,
                                                   self.beta1, self.beta2, self.eps)
        else:
            # Fused masked update (compiled kernel)
            chunk_smoothed = _adam_smooth_masked(self.m, self.v, self.t, self.sel, chunk_detached,
                                                 self.beta1, self.beta2, self.eps)

        # Replace in original tensor
        if self.input_slice is not None:
            output_shape = slice_length(self.input_slice, x.shape[-1])
            if output_shape == x.shape[-1]:
                x_out = chunk_smoothed
            else:
                # only replace a slice, so we need to clone to avoid modifying input
                x_out = x.clone()
                x_out[..., self.input_slice] = chunk_smoothed
        else:
            # we overwrite the whole tensor, no need to clone
            x_out = chunk_smoothed

        return x_out

    def reset(self):
        """Reset the internal state."""
        self.m = torch.tensor(0, dtype=torch.float32)
        self.v = torch.tensor(0, dtype=torch.float32)
        self.t = torch.tensor(0, dtype=torch.int64)

        self.sel = None

    def initialize(self, shape, device) -> None:
        """Initialize the internal state with zeros for the given number of elements and input dimension."""
        self.m = torch.zeros(shape, dtype=torch.float32, device=device)
        self.v = torch.zeros(shape, dtype=torch.float32, device=device)
        self.t = torch.zeros(shape[0], dtype=torch.int64, device=device)

        self.sel = None

    def update_state(self, adam_state: AdamState) -> None:
        """Update the internal state with provided values."""
        m, v, t = adam_state.m, adam_state.v, adam_state.t
        self.m = m
        self.v = v
        self.t = t

        self.sel = None

    def prune(self, prune_mask: Tensor) -> None:
        """Prune the internal state to only keep entries at the specified indices."""
        assert not self.is_reset(), (
            "Cannot prune state that has not been initialized. Call forward() at least once first."
        )
        sel = torch.where(~prune_mask)[0]
        self.m = self.m[sel]
        self.v = self.v[sel]
        self.t = self.t[sel]

        if self.sel is not None:
            self.sel = self.sel[sel]

    def zero_out(self, zero_t=False) -> None:
        """Zero out the moments. Called when resetting gaussians opacities."""
        assert not self.is_reset(), (
            "Cannot zero out state that has not been initialized. Call forward() at least once first."
        )
        self.m = torch.zeros_like(self.m)
        self.v = torch.zeros_like(self.v)
        if zero_t:
            self.t = torch.zeros_like(self.t)

    def replace(self, from_indices: Tensor, dest_indices: Tensor, zero_t=False) -> None:
        """Replace the internal state to duplicate entries at the specified indices."""
        assert not self.is_reset(), (
            "Cannot replace state that has not been initialized. Call forward() at least once first."
        )

        self.m[dest_indices] = self.m[from_indices]
        self.v[dest_indices] = self.v[from_indices]
        if zero_t:
            self.t[dest_indices] = 0
        else:
            self.t[dest_indices] = self.t[from_indices]

    def clone(self, clone_mask: Tensor, zero_t=False) -> None:
        """Clone the internal state to duplicate entries at the specified indices."""
        assert not self.is_reset(), (
            "Cannot extend state that has not been initialized. Call forward() at least once first."
        )

        num_new_rows = clone_mask.sum()
        new_zeros = torch.zeros((num_new_rows, *self.m.shape[1:]), device=self.m.device, dtype=self.m.dtype)
        if zero_t:
            new_t = torch.zeros((num_new_rows, *self.t.shape[1:]), device=self.t.device, dtype=self.t.dtype)
        else:
            sel = torch.where(clone_mask)[0]
            new_t = self.t[sel]

        self.m = torch.cat([self.m, new_zeros], dim=0)
        self.v = torch.cat([self.v, new_zeros], dim=0)
        self.t = torch.cat([self.t, new_t], dim=0)

    def add(self, nr_new: int) -> None:
        """Add new entries to the internal state."""
        assert not self.is_reset(), (
            "Cannot extend state that has not been initialized. Call forward() at least once first."
        )

        new_zeros = torch.zeros((nr_new, *self.m.shape[1:]), device=self.m.device, dtype=self.m.dtype)
        new_t = torch.zeros((nr_new, *self.t.shape[1:]), device=self.t.device, dtype=self.t.dtype)

        self.m = torch.cat([self.m, new_zeros], dim=0)
        self.v = torch.cat([self.v, new_zeros], dim=0)
        self.t = torch.cat([self.t, new_t], dim=0)
    
    def split(self, split_mask: Tensor, N: int, zero_t=False) -> None:
        """Split the internal state to duplicate entries at the specified indices."""
        assert not self.is_reset(), (
            "Cannot extend state that has not been initialized. Call forward() at least once first."
        )

        # Count how many new rows we need
        num_new_rows = split_mask.sum() * N

        # Handle t depending on zero_t flag
        if zero_t:
            new_t = torch.zeros((num_new_rows, *self.t.shape[1:]), device=self.t.device, dtype=self.t.dtype)
        else:
            # Carry each parent's timestep to its children, block-repeated (NOT repeat_interleave) to
            # match the canonical 3DGS split order (Gaussian tensors / vanilla.py state_fn / FastGS).
            sel = torch.where(split_mask)[0]
            new_t = self.t[sel].repeat(N)

        rest_sel = torch.where(~split_mask)[0]

        # Preallocate zeros directly for m and v
        new_zeros = torch.zeros((num_new_rows, *self.m.shape[1:]), device=self.m.device, dtype=self.m.dtype)
        self.m = torch.cat([self.m[rest_sel], new_zeros], dim=0)
        self.v = torch.cat([self.v[rest_sel], new_zeros], dim=0)
        self.t = torch.cat([self.t[rest_sel], new_t], dim=0)

    def get_state(self) -> AdamState:
        """Get the current internal state."""
        return AdamState(self.m, self.v, self.t)

    def subgroups_view(self, slices: dict[str, slice]) -> dict[str, "AdamInputSmoothing"]:
        """
        Create lightweight subgroups that share memory with the main tensor states.

        Args:
            slices (dict[str, slice]): Mapping from subgroup name to slice, e.g.:
                {"means": slice(0, 3) ,"scale": slice(3, 6), "rotation": slice(6, 10), "opacity": slice(10, 11), "sh": slice(11, 59)}

        Returns:
            dict[str, AdamInputSmoothing]: Submodules that share self.m and self.v tensors.
        """
        if not hasattr(self, "m") or self.m.ndim == 0:
            raise RuntimeError("Cannot create subgroups before the first forward() call.")

        subgroups = {}
        for name, slc in slices.items():
            sub = AdamInputSmoothing(
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                input_slice=None
            )

            # share the same memory (not copy)
            sub.m = self.m[..., slc]
            sub.v = self.v[..., slc]
            sub.t = self.t  # shared time step

            subgroups[name] = sub

        return subgroups

    def aggregate_from_subgroups(self, subgroups: dict[str, "AdamInputSmoothing"], slices: dict[str, slice]) -> None:
        """
        Aggregate states from subgroups back into the main module.

        Args:
            subgroups (dict[str, AdamInputSmoothing]): Submodules created via subgroups_view.
            slices (dict[str, slice]): Mapping from subgroup name to slice, e.g.:
                {"means": slice(0, 3) ,"scale": slice(3, 6), "rotation": slice(6, 10), "opacity": slice(10, 11), "sh": slice(11, 59)}
        """
        if not hasattr(self, "m") or self.is_reset():
            raise RuntimeError("Cannot aggregate states before the first forward() call.")

        # Adjust stats shape
        first_m_val = next(iter(subgroups.values())).m
        if self.m.shape[:-1] != first_m_val.shape[:-1]:
            self.m = torch.zeros((*first_m_val.shape[:-1], self.m.shape[-1]), dtype=first_m_val.dtype,
                                 device=first_m_val.device)
            self.v = torch.zeros((*first_m_val.shape[:-1], self.v.shape[-1]), dtype=first_m_val.dtype,
                                 device=first_m_val.device)

        for name, slc in slices.items():
            sub = subgroups[name]
            self.m[..., slc] = sub.m
            self.v[..., slc] = sub.v
        # Assume time step is the same across all subgroups
        self.t = next(iter(subgroups.values())).t

    def is_reset(self) -> bool:
        """Check if the internal state is reset."""
        assert self.m.shape == self.v.shape, "First and second moment vectors must have the same shape."
        return bool(self.m.ndim == 0 and self.v.ndim == 0 and self.t == 0)
