"""Core `Gaussians` dataclass shared across the whole pipeline.

`Gaussians` holds one scene's primitives — means, SH harmonics, opacities, scales, rotations — plus the
optional per-Gaussian fields the optimizer attaches during refinement (gradients, deltas, visibility,
optimization state) and a `nr_valid` count for padded scenes. Helpers (`to`, `clone`, `__getitem__`)
move it across devices and batch dimensions. Initializers produce it, the optimizer refines it, and the
decoder renders it.
"""

from dataclasses import dataclass, fields

from jaxtyping import Float, Bool, Int64, BFloat16
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations_unnorm: Float[Tensor, "batch gaussian 4"]
    rotations: Float[Tensor, "batch gaussian 4"] | None = None
    covariances: Float[Tensor, "batch gaussian dim dim"] | None = None
    probabilities: Float[Tensor, "batch gaussian distr"] | None = None
    # mask: Bool[Tensor, "batch gaussian"] | None = None
    sel: Int64[Tensor, "valid_gaussian_1"] | None = None
    filter_3D: Float[Tensor, "batch gaussian"] | None = None
    gradients: Float[Tensor, "batch valid_gaussian_1 total_dim"] | BFloat16[Tensor, "batch valid_gaussian_1 total_dim"] | None = None
    norm_gradients: Float[Tensor, "batch valid_gaussian_1 total_dim"] | BFloat16[Tensor, "batch valid_gaussian_1 total_dim"] | None = None
    deltas: Float[Tensor, "batch valid_gaussian_2 d_delta"] | BFloat16[Tensor, "batch valid_gaussian_2 d_delta"] | None = None  # In case of predicting scale and mag, the raw deltas are 2*total_dim, cannot use SGD loss directly
    visibility: Float[Tensor, "batch gaussian"] | None = None  # visibility information at the end of the current batch for pruning
    visibility_aggregator: Float[Tensor, "batch gaussian"] | None = None  # aggregates visibility over epoch
    stores_activated: bool = True  # whether scales and opacities are stored in activated form
    nr_valid: int = -1  # the number of valid gaussians (without padding)

    EXCLUDED_FROM_MASKING = {"sel", "stores_activated", "deltas", "gradients", "norm_gradients", "valid_gaussians"}  # deltas are predicted to non masked values
    
    def to(self, device=None, dtype=None) -> "Gaussians":
        """ Move all tensors to the specified device or dtype. """
        def to_with_none(tensor):
            if isinstance(tensor, bool):
                return tensor
            elif isinstance(tensor, int):
                return tensor
            return tensor.to(device=device, dtype=dtype) if tensor is not None else None

        new_tensors = {field.name: to_with_none(getattr(self, field.name)) for field in fields(self)}

        return Gaussians(**new_tensors)

    def clone(self) -> "Gaussians":
        """ Clone all tensors. """
        # handle None and bool fields
        new_tensors = {}
        for field in fields(self):
            tensor = getattr(self, field.name)
            if isinstance(tensor, bool):
                new_tensors[field.name] = tensor
            elif isinstance(tensor, int):
                new_tensors[field.name] = tensor
            elif tensor is not None:
                new_tensors[field.name] = tensor.clone()
            else:
                new_tensors[field.name] = None

        return Gaussians(**new_tensors)

    # Override __getitem__ to support indexing
    def __getitem__(self, idx) -> "Gaussians":
        new_tensors = {}
        for field in fields(self):
            tensor = getattr(self, field.name)
            if isinstance(tensor, bool):
                new_tensors[field.name] = tensor
            elif isinstance(tensor, int):
                new_tensors[field.name] = tensor
            elif tensor is not None and field.name not in self.EXCLUDED_FROM_MASKING:
                new_tensors[field.name] = tensor[idx]
            else:
                new_tensors[field.name] = None
        return Gaussians(**new_tensors)

    def select_valid(self) -> "Gaussians":
        """Return the subset of valid (non-padding) gaussians selected by `sel`.

        Returns self unchanged when `sel` is None.
        """
        if self.sel is None:
            return self
        return self[:, self.sel]

    def sample_subset(self, sampled_indices) -> "Gaussians":
        """ Randomly sample a subset of gaussians. """
        total_gaussians = self.means.shape[1]
        sample_num = len(sampled_indices)

        new_tensors = {}
        for field in fields(self):
            tensor = getattr(self, field.name)
            if tensor is not None:
                if isinstance(tensor, bool):
                    new_tensors[field.name] = tensor
                elif isinstance(tensor, int):
                    new_tensors[field.name] = tensor
                else:
                    new_tensors[field.name] = tensor[:, sampled_indices]
            else:
                new_tensors[field.name] = None
        print(f"Sampled {sample_num} / {total_gaussians} gaussians.")
        return Gaussians(**new_tensors)

    def __len__(self):
        return self.means.shape[1]

    def update_object_by_curr_mask(self, **new_values) -> "Gaussians":
        """ Update certain element using the current mask. """
        sel = self.sel
        new_tensors = {}
        for field in fields(self):
            tensor = getattr(self, field.name)  # [B, G, ...]
            if tensor is not None:
                if field.name in new_values:
                    new_value = new_values[field.name]  # [B, G_valid, ...]
                    if sel is None or new_value is None or field.name in self.EXCLUDED_FROM_MASKING:
                        tensor = new_value
                    else:
                        tensor = tensor.clone()
                        tensor[:, sel, ...] = new_value
                new_tensors[field.name] = tensor
            else:
                if field.name in new_values:
                    if field.name in ["deltas", "gradients", "norm_gradients"]:
                        # Special case: allow updating deltas even if it is None
                        new_tensors[field.name] = new_values[field.name]
                        continue
                    assert new_values[field.name] is None, f"Cannot update a None field! {field.name}, got {new_values[field.name]}"
                new_tensors[field.name] = None
        return Gaussians(**new_tensors)
