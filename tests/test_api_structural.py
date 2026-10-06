"""Tier C: structural / dry-run checks for the Learn2Splat public API (no CUDA, no checkpoint)."""

import inspect
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_import_learn2splat_is_light():
    """`import learn2splat.experimental.api` must not pull torch/hydra; `Learn2Splat` resolves lazily."""
    script = (
        "import sys\n"
        "import learn2splat.experimental.api as api\n"
        "assert 'torch' not in sys.modules, 'torch imported by import learn2splat.experimental.api'\n"
        "assert 'hydra' not in sys.modules, 'hydra imported by import learn2splat.experimental.api'\n"
        "cls = api.Learn2Splat\n"
        "assert cls.__name__ == 'Learn2Splat'\n"
        "assert 'torch' in sys.modules, 'accessing Learn2Splat should load torch'\n"
        "print('OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr[-2000:]!r}"
    assert "OK" in r.stdout


def test_config_bridge_module_defers_heavy_imports():
    script = (
        "import sys\n"
        "import learn2splat.experimental.api.integration.config_bridge as cb\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'hydra' not in sys.modules\n"
        "assert hasattr(cb, 'build_optimizer_cfg') and hasattr(cb, 'load_optimizer_state')\n"
        "print('OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr[-2000:]!r}"
    assert "OK" in r.stdout


def test_learn2splat_error_is_runtimeerror():
    from learn2splat.experimental.api import Learn2SplatError

    assert issubclass(Learn2SplatError, RuntimeError)


def test_public_signatures():
    from learn2splat.experimental.api import Learn2Splat

    sig = inspect.signature(Learn2Splat.__init__)
    params = set(sig.parameters)
    for p in (
        "checkpoint",
        "device",
        "num_refine",
        "iter_batch_size",
        "background_color",
        "strict_load",
    ):
        assert p in params, f"missing Learn2Splat.__init__ param {p}"
    for m in (
        "initialize",
        "optimize",
        "initialize_from_ply",
        "initialize_from_tensors",
        "export_ply",
    ):
        assert callable(getattr(Learn2Splat, m)), f"missing Learn2Splat.{m}"


def test_empty_checkpoint_raises_before_anything():
    from learn2splat.experimental.api import Learn2Splat, Learn2SplatError

    with pytest.raises(Learn2SplatError, match="checkpoint"):
        Learn2Splat(checkpoint="")
    with pytest.raises(Learn2SplatError, match="checkpoint"):
        Learn2Splat(checkpoint=None)  # type: ignore[arg-type]


def test_scene_protocol_accepts_inria_shaped_and_reports_missing():
    from learn2splat.experimental.api.integration.scene_protocol import (
        Learn2SplatError,
        assert_scene_protocol,
    )

    cam = types.SimpleNamespace(
        R=0, T=0, FoVx=1.0, FoVy=1.0, image_width=64, image_height=48,
        original_image=0,
    )
    gm = types.SimpleNamespace(
        active_sh_degree=0, max_sh_degree=3,
        _xyz=0, _features_dc=0, _features_rest=0,
        _scaling=0, _rotation=0, _opacity=0,
        save_ply=lambda p: None, load_ply=lambda p: None,
    )
    scene = types.SimpleNamespace(
        cameras_extent=1.0, gaussians=gm, getTrainCameras=lambda scale=1.0: [cam]
    )
    assert_scene_protocol(scene)  # inria-shaped duck object passes

    bad = types.SimpleNamespace(
        cameras_extent=1.0, gaussians=gm, getTrainCameras=lambda scale=1.0: [cam]
    )
    del bad.cameras_extent
    with pytest.raises(Learn2SplatError, match="cameras_extent"):
        assert_scene_protocol(bad)

    gm2 = types.SimpleNamespace(
        active_sh_degree=0, max_sh_degree=3, _xyz=0, _features_dc=0,
        _features_rest=0, _scaling=0, _rotation=0, _opacity=0,
        save_ply=lambda p: None,  # missing load_ply
    )
    scene2 = types.SimpleNamespace(
        cameras_extent=1.0, gaussians=gm2, getTrainCameras=lambda scale=1.0: [cam]
    )
    with pytest.raises(Learn2SplatError, match="load_ply"):
        assert_scene_protocol(scene2)


def test_feature_conditioned_checkpoint_warns_and_proceeds(monkeypatch, tmp_path):
    """init_state_wo_features=False warns, then proceeds with a standard-normal state init."""
    import learn2splat.config as cfgmod
    import learn2splat.experimental.api.integration.config_bridge as cb
    import learn2splat.misc.hf_ckpt as hf

    fake_cfg_path = tmp_path / "config.yaml"
    fake_cfg_path.write_text("placeholder: true\n")

    monkeypatch.setattr(hf, "maybe_resolve_hf_ref", lambda p: p)
    monkeypatch.setattr(cfgmod, "_find_config_for_checkpoint", lambda p: fake_cfg_path)

    opt_cfg = types.SimpleNamespace(
        init_state_wo_features=False, init_state_type="constant", init_state_scale=0,
        name="knn_based",
    )
    monkeypatch.setattr(cb, "build_optimizer_cfg", lambda p: (opt_cfg, 100))

    # build_optimizer is now reached (the feature gate warns instead of erroring);
    # stop the constructor there to avoid the CUDA path.
    class _Stop(Exception):
        pass

    def _stop(cfg):
        raise _Stop

    monkeypatch.setattr(cb, "build_optimizer", _stop)

    from learn2splat.experimental.api import Learn2Splat

    with pytest.warns(UserWarning, match="init_state_wo_features"):
        with pytest.raises(_Stop):
            Learn2Splat(checkpoint="/tmp/whatever.ckpt", device="cuda")

    # the feature gate flipped the cfg: feature-free + standard-normal state init
    assert opt_cfg.init_state_wo_features is True
    assert opt_cfg.init_state_type == "random"
    assert opt_cfg.init_state_scale == 1.0


def test_missing_config_yaml_raises(monkeypatch):
    import learn2splat.config as cfgmod
    import learn2splat.misc.hf_ckpt as hf

    monkeypatch.setattr(hf, "maybe_resolve_hf_ref", lambda p: p)
    monkeypatch.setattr(cfgmod, "_find_config_for_checkpoint", lambda p: None)

    from learn2splat.experimental.api import Learn2Splat, Learn2SplatError

    with pytest.raises(Learn2SplatError, match="no config.yaml"):
        Learn2Splat(checkpoint="/tmp/whatever.ckpt", device="cuda")


def test_resplat_official_config_built_from_config_group(monkeypatch, tmp_path):
    """Official ReSplat .pth files carry no config; the architecture comes from resplat_v2_<size>.

    The API (and demo.py) rebuild a learned initializer from a checkpoint's config.yaml. ReSplat's
    released checkpoints have none, so the resplat_v2_<size> config group is composed instead —
    keeping the groups the only definition of that architecture.
    """
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from learn2splat.config import resplat_official_config_path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # keep the real cache clean
    GlobalHydra.instance().clear()

    for size, vit, channels in [("small", "vits", 256), ("base", "vitb", 512), ("large", "vitl", 768)]:
        path = resplat_official_config_path(f"/tmp/resplat-{size}-dl3dv-256x448-view8-abc1234.pth")
        assert path is not None and path.exists(), size
        cfg = OmegaConf.load(path)
        si, so = cfg.scene_trainer.scene_initializer, cfg.scene_trainer.scene_optimizer
        assert (si.monodepth_vit_type, si.gaussian_regressor_channels) == (vit, channels)
        assert si.num_blocks == 6
        # ReSplat predicts rotations in camera space; both sides must agree.
        assert si.camera_frame_quats is True and so.camera_frame_quats is True
        assert so.name == "resplat_v2" and so.input_error_mv_attn_blocks == 1
        assert cfg.scene_trainer.decoder.eps2d == 0.1


def test_resplat_official_config_ignores_other_checkpoints(tmp_path, monkeypatch):
    """Only official ReSplat .pth files get the generated config; everything else resolves normally."""
    from learn2splat.config import resplat_official_config_path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resplat_official_config_path("/tmp/epoch_7-step_150000.ckpt") is None
    assert resplat_official_config_path("/tmp/gmdepth-scale1-resumeflowthings.pth") is None
