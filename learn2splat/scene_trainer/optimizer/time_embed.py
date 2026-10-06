# From: https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py

import math

import torch


def get_ddpm_sinusoidal_time_embedding(timesteps, embedding_dim):
    """DDPM/Transformer sinusoidal timestep embedding, as used by G3R/QuickSplat (their
    `models/time_embedding.py`). `timesteps` is already normalized upstream (t / horizon), so this only
    turns the scalar step into a [N, embedding_dim] sinusoidal vector; 10000 is the frequency max-period
    (NOT a normalizer). This is the G3R-faithful alternative to the NeRF-style `get_embedder` below,
    selected via `time_encoding_type=ddpm`."""
    assert timesteps.ndim == 1, f"expected 1D timesteps, got {timesteps.shape}"
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero-pad the last dim when embedding_dim is odd
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,  # time steps are 1D
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class TimeEncodingWrapper:
    def __init__(self, use_time_encoding, time_encoder_fn, t, T, state):
        self.use_time_encoding = use_time_encoding
        self.T = T
        self.time_encoder_fn = time_encoder_fn
        self.state = state
        self.t = t

    def __enter__(self):
        # We are modifying the state only inside the context manager
        state = self.state
        if self.use_time_encoding:
            assert self.time_encoder_fn is not None, "Time encoder function must be defined."

            rel_step = torch.tensor([self.t / self.T], device=state.device)

            time_encoding = self.time_encoder_fn(rel_step)  # [embedding_dim]
            time_encoding = time_encoding.unsqueeze(0).repeat(state.shape[0], 1)  # [N, embedding_dim]

            # Concatenate encoding to state
            state = torch.cat([state, time_encoding], dim=-1)  # [N, c+embedding_dim]

        return state  # returns the modified state

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do nothing, the original state is preserved outside the context manager
        # Return False to propagate exceptions, if any
        return False


if __name__ == "__main__":
    # Example usage
    embed_fn, output_dim = get_embedder(multires=6)
    print(f"Output embedding dimension: {output_dim}")
    steps = torch.randn(10, 1)  # Example input (steps normalized between 0 and 1)
    print(f"Input shape: {steps.shape}")
    print("steps[0:2]:", steps[0:2])
    embedded_x = embed_fn(steps)
    print(f"Embedded shape: {embedded_x.shape}")
    print("embedded_x[0:2]:", embedded_x[0:2])
