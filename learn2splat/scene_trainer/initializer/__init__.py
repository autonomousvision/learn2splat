from .initializer import Initializer
from .initializer_resplat import ResplatInitializer, ResplatInitializerCfg
from .initializer_colmap import InitializerColmap, InitializerColmapCfg
from .initializer_ply import InitializerPly, InitializerPlyCfg
from .initializer_edgs import InitializerEdgs, InitializerEdgsCfg
from .initializer_random import InitializerRandom, InitializerRandomCfg
from .initializer_pointcloud import InitializerPointcloud, InitializerPointcloudCfg
from .initializer_depth import InitializerDepth, InitializerDepthCfg

SCENE_INITIALIZERS = {
    "resplat_v1": ResplatInitializer,
    "resplat_v2": ResplatInitializer,
    "colmap": InitializerColmap,
    "ply": InitializerPly,
    "edgs": InitializerEdgs,
    "random": InitializerRandom,
    "pointcloud": InitializerPointcloud,
    "depth": InitializerDepth,
}

InitializerCfg = ResplatInitializerCfg | InitializerColmapCfg | InitializerPlyCfg | InitializerEdgsCfg | InitializerRandomCfg | InitializerPointcloudCfg | InitializerDepthCfg


def get_scene_initializer(cfg: InitializerCfg) -> Initializer:
    print(f"Using scene initializer: {cfg.name}")
    scene_initializer = SCENE_INITIALIZERS[cfg.name]
    scene_initializer = scene_initializer(cfg)
    return scene_initializer
