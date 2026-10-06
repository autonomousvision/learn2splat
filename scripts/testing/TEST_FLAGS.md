# Test output flags

In `mode=test`, the outputs written per scene are controlled by `meta_trainer.test.*` flags
(defaults in [`learn2splat/config/main.yaml`](../../learn2splat/config/main.yaml)). Pass them as Hydra overrides, e.g.
`meta_trainer.test.save_render_image=true`. Everything is written under `<output_dir>/`:

- images — `<output_dir>/<module>/images/<scene>/`
- Gaussians / poses — `<output_dir>/<module>/`
- cameras — `<output_dir>/cameras/`
- metrics — `<output_dir>/metrics/` and per-scene `<output_dir>/<module>/metrics/<scene>/`
  (see [METRICS.md](METRICS.md) for the file layout, shapes, and partial-scene handling, and
  [TIMINGS.md](TIMINGS.md) for the runtime/memory numbers and GPU warm-up)

**Scoring / evaluation**

| Flag | Default | Effect |
|---|---|---|
| `compute_scores` | `true` | Compute and save PSNR/SSIM/LPIPS metrics. |
| `compute_scores_metrics` | `[psnr,ssim,lpips]` | Which metrics to compute. |
| `metrics_batch_size` | `32` | Batch size for metric computation. |
| `eval_initialization` | `true` | Also evaluate/save the initializer output (before optimization). |
| `eval_context_views` | `false` | Also evaluate/save the context (input) views, not just target. |

**Images**

| Flag | Default | Effect |
|---|---|---|
| `save_render_image` | `false` | Save the rendered RGB trajectory per view (`color_<view>/`). |
| `save_render_image_last_only` | `false` | Save only the final iteration's render. |
| `save_gt_image` | `false` | Save GT RGB next to the final render (`last/color_<view>/<idx>_gt.png`). |
| `save_error_image` | `false` | Save RGB error maps (`error_<view>/`). |

**Depth** (rendered = rasterized from the Gaussians; pred = the initializer's input-view estimate)

| Flag | Default | Effect |
|---|---|---|
| `save_render_depth` | `false` | Save rendered depth → `rendered_depth_<view>/`. |
| `save_gt_depth` | `false` | Save dataset GT depth next to it (when the dataset provides depth). |
| `save_init_pred_depth` | `false` | Save the initializer's predicted input-view depth → `pred_depth_context/`. |

**Video** (`save_video` is a master switch — enable at least one mode below)

| Flag | Default | Effect |
|---|---|---|
| `save_video` | `false` | Master switch for videos (`videos/`); enable at least one mode below. Shared: `save_video_view_index`, `save_video_frame_repeat`. |
| `save_video_optim` | `false` | Optimization progress at a fixed view (one viewpoint across the saved steps). |
| `save_video_orbit` | `false` | Camera orbit around the scene at the chosen step(s); `save_video_orbit_steps`, `save_video_orbit_with_optim`. |
| `save_video_optim_orbit` | `false` | Combined: optimization progress with orbits spliced in; `save_video_optim_orbit_steps`, `save_video_orbit_span`. |
| `save_video_orbit_sweeps` / `save_video_orbit_frame_repeat` | `3` / `3` | How long each spliced sweep plays: forward+backward passes over the path, and frames each rendered frame is held for. |
| `save_video_orbit_fitted_path` | `false` | Sweep a turntable arc around the subject instead of the views in index order (for DTU / dense colmap scenes, whose view order is not a camera path). Starts at `save_video_view_index` and turns about the scene's vertical, so it leaves and rejoins that view without a jump; `save_video_orbit_fitted_path_frames` / `..._span_deg` set how far it turns and how slowly. |

**Gaussians / cameras**

| Flag | Default | Effect |
|---|---|---|
| `save_gaussian` | `false` | Save Gaussian `.ply` per saved step (`gaussians/`). |
| `save_gaussian_last_only` | `false` | With `save_gaussian`, write only the final step's `.ply` instead of one per saved step. |
| `save_poses` | `false` | Save camera poses JSON (`poses/`). |
| `save_cameras_json` / `save_cameras_npz` | `true` | Save renderer-ready cameras (`cameras/`). |

**Which iterations are saved**

| Flag | Default | Effect |
|---|---|---|
| `save_every_freq` / `save_every_steps` | `null` | Paired lists: save frequency per step range (same length). |
| `save_at_iters` | `null` | Explicit list of iterations to save. |

**Postprocessing** — `meta_trainer/test/postprocessing=<name>` selects an extra refinement applied after
the learned optimizer (e.g., `none`, `adam`, `sgd`).

**Other**

| Flag | Default | Effect |
|---|---|---|
| `skip_if_outputs_exist` | `false` | Skip scenes whose outputs already exist. |
| `scenes_filter` | `null` | Restrict evaluation to a list of scene names. |
| `render_chunk_size` | `null` | Views per chunk when rendering the video camera path. Videos only — for the init / saved-step renders (the ones that rasterize the full view set at once, and the usual source of a rasterizer OOM) set `scene_trainer.iter_batch_size` instead. |
| `inference_window_size` | `null` | Sliding window over input views at inference. |
| `stabilize_camera` / `stab_camera_kernel` | `false` / `50` | Smooth camera trajectory for video. |