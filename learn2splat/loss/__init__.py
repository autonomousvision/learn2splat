from .loss import Loss
from .loss_deltas import LossDeltas, LossDeltasCfgWrapper
from .loss_iso_scales import LossIsoScalesCfgWrapper, LossIsoScales
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_render_depth import LossRenderDepth, LossRenderDepthCfgWrapper
from .loss_sh0 import LossSh0CfgWrapper, LossSh0
from .loss_ssim import LossSsimCfgWrapper, LossSsim
from .loss_sgd import LossSGDCfgWrapper, LossSGD
from .loss_gaussians import LossGaussiansCfgWrapper, LossGaussians
from .loss_stability import LossStabilityCfgWrapper, LossStability

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDeltasCfgWrapper: LossDeltas,
    LossSsimCfgWrapper: LossSsim,
    LossSh0CfgWrapper: LossSh0,
    LossIsoScalesCfgWrapper: LossIsoScales,
    LossSGDCfgWrapper: LossSGD,
    LossGaussiansCfgWrapper: LossGaussians,
    LossRenderDepthCfgWrapper: LossRenderDepth,
    LossStabilityCfgWrapper: LossStability,
}

LossCfgWrapper = (
        LossLpipsCfgWrapper |
        LossMseCfgWrapper |
        LossDeltasCfgWrapper |
        LossSsimCfgWrapper |
        LossSh0CfgWrapper |
        LossIsoScalesCfgWrapper |
        LossSGDCfgWrapper |
        LossGaussiansCfgWrapper |
        LossRenderDepthCfgWrapper |
        LossStabilityCfgWrapper
)


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
