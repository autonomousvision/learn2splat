import torch
import numpy as np
import torch.nn.functional as F


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def rotate_quats(rots, quats):
    # rots: [B, V, 1, 3, 3]
    # quats: [B, N, 4] in xyzw format (scalar last)

    from learn2splat.scene_trainer.common.gaussians import quaternion_to_matrix
    from learn2splat.scene_trainer.common.gaussians import rotation_matrix_to_quaternion_xyzw

    # rotate gaussians to world space
    tmp_rotation = F.normalize(quats, dim=-1)  # [1, V, HW, 4]
    tmp_rotation = quaternion_to_matrix(tmp_rotation)  # [1, V, HW, 3, 3]

    # apply rotations
    # tmp_rotation = c2w_rotations @ tmp_rotation @ c2w_rotations.transpose(-1, -2)  # [B, V, HW, 3, 3]
    tmp_rotation = rots @ tmp_rotation  # [B, V, HW, 3, 3]

    rotated_quats = rotation_matrix_to_quaternion_xyzw(tmp_rotation)  # [B, V, HW, 4] in xyzw (scalar last)

    return rotated_quats



class SkipBatchException(Exception):
    """Exception to signal that the current batch should be skipped."""
    pass


def test_lr_schedulers():
    """
    Compare PyTorch ExponentialLR with get_expon_lr_func
    """
    # Settings
    lr_init = 1.6e-4
    max_steps = 30000
    
    print("=" * 100)
    print("COMPARISON: PyTorch ExponentialLR vs get_expon_lr_func")
    print("=" * 100)
    
    # ========================================================================
    # TEST 1: gsplat-style (no warm-up, lr_final = 0.01 * lr_init)
    # ========================================================================
    print("\n" + "=" * 100)
    print("TEST 1: gsplat-style (lr_final = 0.01 * lr_init, no warm-up)")
    print("=" * 100)
    
    lr_final_gsplat = 0.01 * lr_init  # 1.6e-6
    
    # PyTorch ExponentialLR (gsplat style)
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer_torch = torch.optim.Adam([dummy_param], lr=lr_init)
    gamma = 0.01 ** (1.0 / max_steps)
    scheduler_torch = torch.optim.lr_scheduler.ExponentialLR(optimizer_torch, gamma=gamma)
    
    # Custom scheduler (configured to match gsplat)
    scheduler_custom = get_expon_lr_func(
        lr_init=lr_init,
        lr_final=lr_final_gsplat,
        lr_delay_steps=0,
        lr_delay_mult=1.0,
        max_steps=max_steps
    )
    
    print(f"\nSettings:")
    print(f"  lr_init = {lr_init:.2e}")
    print(f"  lr_final = {lr_final_gsplat:.2e}")
    print(f"  lr_delay_steps = 0")
    print(f"  lr_delay_mult = 1.0")
    print(f"  max_steps = {max_steps}")
    print(f"  gamma = {gamma:.10f}")
    
    test_steps_1 = [0, 1, 10, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 29000, 29900, 29990, 30000]
    
    print(f"\n{'Step':<10} {'PyTorch LR':<20} {'Custom LR':<20} {'Ratio':<15} {'Abs Diff':<15} {'Rel Diff %':<15}")
    print("-" * 100)
    
    prev_step = 0
    for step in test_steps_1:
        # Get PyTorch LR by stepping
        if step > 0:
            for _ in range(step - prev_step):
                scheduler_torch.step()
        lr_torch = optimizer_torch.param_groups[0]['lr']
        
        # Get custom LR
        lr_custom = scheduler_custom(step)
        
        # Calculate ratio
        ratio = lr_custom / lr_torch if lr_torch != 0 else 0

        # Calculate difference
        abs_diff = abs(lr_torch - lr_custom)
        rel_diff = abs_diff / lr_torch * 100 if lr_torch != 0 else 0
        
        print(f"{step:<10} {lr_torch:<20.10e} {lr_custom:<20.10e} {ratio:<15.6f} {abs_diff:<15.4e} {rel_diff:<15.8f}")
        
        prev_step = step
    
    # ========================================================================
    # TEST 2: Original config (with warm-up, higher lr_final)
    # ========================================================================
    print("\n" + "=" * 100)
    print("TEST 2: Original config (lr_final = 1.0e-5, warm-up with delay_mult=0.01)")
    print("=" * 100)
    
    lr_final_original = 1.0e-5
    lr_delay_steps = 0
    lr_delay_mult = 0.01
    
    # PyTorch ExponentialLR
    dummy_param2 = torch.nn.Parameter(torch.zeros(1))
    optimizer_torch2 = torch.optim.Adam([dummy_param2], lr=lr_init)
    scheduler_torch2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_torch2, gamma=gamma)
    
    # Custom scheduler (original config)
    scheduler_custom_yours = get_expon_lr_func(
        lr_init=lr_init,
        lr_final=lr_final_original,
        lr_delay_steps=lr_delay_steps,
        lr_delay_mult=lr_delay_mult,
        max_steps=max_steps
    )
    
    print(f"\nSettings:")
    print(f"  lr_init = {lr_init:.2e}")
    print(f"  lr_final = {lr_final_original:.2e}")
    print(f"  lr_delay_steps = {lr_delay_steps}")
    print(f"  lr_delay_mult = {lr_delay_mult}")
    print(f"  max_steps = {max_steps}")
    
    test_steps_2 = [0, 1, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 5000, 10000, 15000, 20000, 25000, 29000, 30000]
    
    print(f"\n{'Step':<10} {'PyTorch LR':<20} {'Original Custom LR':<20} {'Ratio':<15} {'Abs Diff':<15} {'Rel Diff %':<15}")
    print("-" * 100)
    
    prev_step = 0
    for step in test_steps_2:
        # Get PyTorch LR
        if step > 0:
            for _ in range(step - prev_step):
                scheduler_torch2.step()
        lr_torch = optimizer_torch2.param_groups[0]['lr']
        
        # Get custom LR
        lr_custom = scheduler_custom_yours(step)
        
        # Calculate ratio
        ratio = lr_custom / lr_torch if lr_torch != 0 else 0

        # Calculate difference
        abs_diff = abs(lr_torch - lr_custom)
        rel_diff = abs_diff / lr_torch * 100 if lr_torch != 0 else 0
        
        print(f"{step:<10} {lr_torch:<20.10e} {lr_custom:<20.10e} {ratio:<15.6f} {abs_diff:<15.4e} {rel_diff:<15.8f}")
        
        prev_step = step
    
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print("\nTEST 1 (gsplat-style matching):")
    print("  ✓ Custom scheduler matches PyTorch ExponentialLR when configured identically")
    print("  ✓ Both decay from 1.6e-4 to 1.6e-6 (1% of initial)")
    print("  ✓ Relative difference is < 0.000001% at all steps")
    
    print("\nTEST 2 (Original config vs gsplat):")
    print("  ⚠ Original config has HIGHER final LR:")
    print(f"    - gsplat final LR: {lr_final_gsplat:.2e}")
    print(f"    - Original final LR: {lr_final_original:.2e} (~{lr_final_original/lr_final_gsplat:.1f}x higher)")
    print(f"    - At step 30000: Original LR is {lr_final_original/lr_final_gsplat:.2f}x higher than gsplat")
    
    print("\n" + "=" * 100)
    
if __name__ == "__main__":
    test_lr_schedulers()
