from ...dataset import DatasetCfg
from .decoder import Decoder
from .gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg, GSplatDecoderSplattingCUDA

DECODERS = {
    "gsplat": GSplatDecoderSplattingCUDA,
}

# name -> Cfg dataclass, for resolving the discriminated union by `name` at the
# top level: dacite's from_dict can't take the `DecoderCfg` union directly (a
# union isn't a class), so callers parsing a raw config look the variant up here.
DECODER_CFGS = {
    "gsplat": GSplatDecoderSplattingCUDACfg,
}

DecoderCfg = GSplatDecoderSplattingCUDACfg

# The inria and fastgs decoders are optional (each needs its own CUDA
# rasterizer backend). Importing this package must NOT require either — gsplat
# is the default. If one is requested while its backend is missing, raise a
# clear, chained ImportError (mirrors the RoMa handling in
# learn2splat/experimental/edgs/init.py) instead of silently degrading.
try:
    from .decoder_splatting_cuda import InriaDecoderSplattingCUDACfg, InriaDecoderSplattingCUDA
    DECODERS["inria"] = InriaDecoderSplattingCUDA
    DECODER_CFGS["inria"] = InriaDecoderSplattingCUDACfg
    DecoderCfg = DecoderCfg | InriaDecoderSplattingCUDACfg
except ImportError as _e:
    # `except ... as _e` is auto-deleted at block end; keep a stable ref so the
    # closure below can chain from the original error.
    _INRIA_IMPORT_ERROR = _e

    def _inria_decoder_unavailable(*_args, **_kwargs):
        raise ImportError(
            "The inria decoder requires diff_gaussian_rasterization, which is "
            "not installed. Install it with: "
            "pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
        ) from _INRIA_IMPORT_ERROR

    DECODERS["inria"] = _inria_decoder_unavailable

try:
    from .fastgs_decoder_splatting_cuda import FastGSDecoderSplattingCUDACfg, FastGSDecoderSplattingCUDA
    DECODERS["fastgs"] = FastGSDecoderSplattingCUDA
    DECODER_CFGS["fastgs"] = FastGSDecoderSplattingCUDACfg
    DecoderCfg = DecoderCfg | FastGSDecoderSplattingCUDACfg
except ImportError as _e:
    _FASTGS_IMPORT_ERROR = _e

    def _fastgs_decoder_unavailable(*_args, **_kwargs):
        raise ImportError(
            "The fastgs decoder requires diff_gaussian_rasterization_fastgs, "
            "which is not installed. Build it with: "
            "WITH_OPTIONAL_RASTERIZERS=1 bash setup.sh"
        ) from _FASTGS_IMPORT_ERROR

    DECODERS["fastgs"] = _fastgs_decoder_unavailable


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    print(f"Using decoder: {decoder_cfg.name}")
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
