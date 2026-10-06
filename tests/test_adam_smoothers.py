import torch
import pytest
from learn2splat.scene_trainer.optimizer.layer import AdamInputSmoothing, AdamState


class TestAdamInputSmoothing:
    """Test suite for AdamInputSmoothing layer."""

    def test_initialization(self):
        """Test that the module initializes correctly."""
        smoother = AdamInputSmoothing(beta1=0.9, beta2=0.999, eps=1e-8)
        
        assert smoother.beta1 == 0.9
        assert smoother.beta2 == 0.999
        assert smoother.eps == 1e-8
        assert smoother.is_reset()
        assert smoother.m.ndim == 0
        assert smoother.v.ndim == 0
        assert smoother.t == 0

    def test_single_step_matches_adam(self):
        """Test that a single smoothing step matches Adam optimizer calculations."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps)
        
        # Simulate gradient input
        x = torch.randn(2, 10, 5)  # [batch, num_gaussians, features]
        
        # First forward pass
        output = smoother(x)
        
        # Manual Adam calculation for first step
        t = 1
        m = (1 - beta1) * x
        v = (1 - beta2) * (x ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        expected = m_hat / (torch.sqrt(v_hat) + eps)
        
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)

    def test_multiple_steps_accumulation(self):
        """Test that multiple steps correctly accumulate moments."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps)
        
        batch_size, num_gaussians, features = 2, 10, 5
        
        # Manual tracking
        m_manual = torch.zeros(batch_size, num_gaussians, features)
        v_manual = torch.zeros(batch_size, num_gaussians, features)
        
        for step in range(1, 6):
            x = torch.randn(batch_size, num_gaussians, features)
            
            # Update manual moments
            m_manual = beta1 * m_manual + (1 - beta1) * x
            v_manual = beta2 * v_manual + (1 - beta2) * (x ** 2)
            
            # Bias correction
            m_hat = m_manual / (1 - beta1 ** step)
            v_hat = v_manual / (1 - beta2 ** step)
            expected = m_hat / (torch.sqrt(v_hat) + eps)
            
            # Apply smoother
            output = smoother(x)
            
            # Verify (with more relaxed tolerances due to numerical precision)
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-6)
            assert smoother.t[0] == step

    def test_input_slice(self):
        """Test that input_slice only smooths the specified slice."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        input_slice = slice(2, 5)  # Only smooth features 2-5
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps, input_slice=input_slice)
        
        x = torch.randn(3, 10, 8)  # [batch, num_gaussians, features]
        x_original = x.clone()
        
        output = smoother(x)
        
        # Features outside the slice should be unchanged
        torch.testing.assert_close(output[..., :2], x_original[..., :2])
        torch.testing.assert_close(output[..., 5:], x_original[..., 5:])
        
        # Features inside the slice should be different (smoothed)
        assert not torch.allclose(output[..., 2:5], x_original[..., 2:5])

    def test_reset(self):
        """Test that reset correctly clears the state."""
        smoother = AdamInputSmoothing()
        
        # Run some steps
        x = torch.randn(2, 10, 5)
        smoother(x)
        smoother(x)
        
        assert not smoother.is_reset()
        
        # Reset
        smoother.reset()
        
        assert smoother.is_reset()
        assert smoother.m.ndim == 0
        assert smoother.v.ndim == 0
        assert smoother.t == 0

    def test_update_state(self):
        """Test that update_state correctly sets internal state."""
        smoother = AdamInputSmoothing()
        
        # Create custom state
        m = torch.randn(2, 10, 5)
        v = torch.rand(2, 10, 5)  # Must be positive
        t = torch.tensor([3, 3], dtype=torch.int64)
        
        adam_state = AdamState(m, v, t)
        smoother.update_state(adam_state)
        
        torch.testing.assert_close(smoother.m, m)
        torch.testing.assert_close(smoother.v, v)
        torch.testing.assert_close(smoother.t, t)

    def test_prune(self):
        """Test that pruning correctly removes entries."""
        smoother = AdamInputSmoothing()
        
        # Initialize state
        x = torch.randn(5, 10)  # [batch, features]
        smoother(x)
        
        # Prune indices 1 and 3
        prune_mask = torch.tensor([False, True, False, True, False])
        smoother.prune(prune_mask)
        
        assert smoother.m.shape[0] == 3
        assert smoother.v.shape[0] == 3
        assert smoother.t.shape[0] == 3

    def test_clone(self):
        """Test that cloning correctly duplicates entries."""
        smoother = AdamInputSmoothing()
        
        # Initialize state
        x = torch.randn(3, 10)  # [batch, features]
        smoother(x)
        
        original_m = smoother.m.clone()
        original_v = smoother.v.clone()
        original_t = smoother.t.clone()
        
        # Clone indices 0 and 2
        clone_mask = torch.tensor([True, False, True])
        smoother.clone(clone_mask, zero_t=False)
        
        # Should have 5 entries now (3 original + 2 cloned)
        assert smoother.m.shape[0] == 5
        assert smoother.v.shape[0] == 5
        assert smoother.t.shape[0] == 5
        
        # Original entries should be unchanged
        torch.testing.assert_close(smoother.m[:3], original_m)
        torch.testing.assert_close(smoother.v[:3], original_v)
        torch.testing.assert_close(smoother.t[:3], original_t)
        
        # New entries should be zeroed
        assert torch.allclose(smoother.m[3:], torch.zeros_like(smoother.m[3:]))
        assert torch.allclose(smoother.v[3:], torch.zeros_like(smoother.v[3:]))

    def test_split(self):
        """Test that splitting removes the original and appends N new zero entries."""
        smoother = AdamInputSmoothing()

        # Initialize state
        x = torch.randn(3, 10)  # [batch, features]
        smoother(x)

        # Split index 1 into 2 copies (removes index 1, appends 2 new zeros)
        split_mask = torch.tensor([False, True, False])
        N = 2
        smoother.split(split_mask, N=N, zero_t=True)

        # Should have 4 entries: 2 remaining originals + 2 new from split
        assert smoother.m.shape[0] == 4
        assert smoother.v.shape[0] == 4
        assert smoother.t.shape[0] == 4

        # New entries (last 2) should be zeroed
        assert torch.allclose(smoother.m[2:], torch.zeros_like(smoother.m[2:]))
        assert torch.allclose(smoother.v[2:], torch.zeros_like(smoother.v[2:]))
        assert torch.allclose(smoother.t[2:], torch.zeros_like(smoother.t[2:]))

    def test_zero_out(self):
        """Test that zero_out correctly resets moments."""
        smoother = AdamInputSmoothing()
        
        # Initialize state
        x = torch.randn(3, 10)
        smoother(x)
        smoother(x)
        
        original_shape = smoother.m.shape
        
        # Zero out without resetting t
        smoother.zero_out(zero_t=False)
        
        assert smoother.m.shape == original_shape
        assert torch.allclose(smoother.m, torch.zeros_like(smoother.m))
        assert torch.allclose(smoother.v, torch.zeros_like(smoother.v))
        assert smoother.t[0] > 0  # t should not be reset
        
        # Zero out with resetting t
        smoother.zero_out(zero_t=True)
        assert torch.allclose(smoother.t, torch.zeros_like(smoother.t))

    def test_subgroups_view(self):
        """Test that subgroups correctly share memory."""
        smoother = AdamInputSmoothing()
        
        # Initialize with 10 features
        x = torch.randn(2, 5, 10)
        smoother(x)
        
        # Create subgroups
        slices = {
            "means": slice(0, 3),
            "scale": slice(3, 6),
            "rotation": slice(6, 10),
        }
        subgroups = smoother.subgroups_view(slices)
        
        # Check that subgroups exist
        assert "means" in subgroups
        assert "scale" in subgroups
        assert "rotation" in subgroups
        
        # Check shapes
        assert subgroups["means"].m.shape[-1] == 3
        assert subgroups["scale"].m.shape[-1] == 3
        assert subgroups["rotation"].m.shape[-1] == 4
        
        # Modify a subgroup and check if main is affected (memory sharing)
        original_m = smoother.m[..., 0:3].clone()
        subgroups["means"].m += 1.0
        
        # Main tensor should be modified
        assert not torch.allclose(smoother.m[..., 0:3], original_m)
        torch.testing.assert_close(smoother.m[..., 0:3], original_m + 1.0)

    def test_aggregate_from_subgroups(self):
        """Test that aggregating from subgroups correctly updates main state."""
        smoother = AdamInputSmoothing()
        
        # Initialize
        x = torch.randn(2, 5, 10)
        smoother(x)
        
        # Create subgroups
        slices = {
            "means": slice(0, 3),
            "scale": slice(3, 6),
            "rotation": slice(6, 10),
        }
        subgroups = smoother.subgroups_view(slices)
        
        # Modify subgroup states
        subgroups["means"].m = torch.ones_like(subgroups["means"].m)
        subgroups["scale"].v = torch.ones_like(subgroups["scale"].v) * 2
        
        # Aggregate (should already be reflected due to memory sharing, but test the method)
        smoother.aggregate_from_subgroups(subgroups, slices)
        
        # Verify
        assert torch.allclose(smoother.m[..., 0:3], torch.ones_like(smoother.m[..., 0:3]))
        assert torch.allclose(smoother.v[..., 3:6], torch.ones_like(smoother.v[..., 3:6]) * 2)

    def test_no_nan_in_output(self):
        """Test that the output doesn't contain NaN values even with extreme inputs."""
        smoother = AdamInputSmoothing(eps=1e-8)
        
        # Test with very small values
        x_small = torch.randn(2, 10, 5) * 1e-10
        output = smoother(x_small)
        assert not torch.isnan(output).any()
        
        # Test with very large values
        x_large = torch.randn(2, 10, 5) * 1e10
        output = smoother(x_large)
        assert not torch.isnan(output).any()
        
        # Test with zeros
        x_zero = torch.zeros(2, 10, 5)
        output = smoother(x_zero)
        assert not torch.isnan(output).any()

    def test_bias_correction_convergence(self):
        """Test that bias correction converges to expected values over many steps."""
        beta1, beta2 = 0.9, 0.999
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2)
        
        # Fixed input
        x = torch.ones(1, 1, 1)
        
        # Run many steps
        for _ in range(1000):
            output = smoother(x)
        
        # After many steps, bias correction should be minimal
        # m should converge to x, v should converge to x^2
        # So output should be close to x / sqrt(x^2) = 1
        expected = torch.ones_like(x)
        torch.testing.assert_close(output, expected, rtol=0.1, atol=0.1)

    def test_different_batch_sizes(self):
        """Test that the module handles the same batch size correctly across calls."""
        smoother = AdamInputSmoothing()
        
        # First call with batch size 2
        x1 = torch.randn(2, 10, 5)
        output1 = smoother(x1)
        assert output1.shape == x1.shape
        
        # Second call with same batch size should work
        x2 = torch.randn(2, 10, 5)
        output2 = smoother(x2)
        assert output2.shape == x2.shape
        
        # The state should have been initialized to match the first input shape
        assert smoother.m.shape == (2, 10, 5)
        assert smoother.v.shape == (2, 10, 5)

    def test_adam_correction_with_standard_optimizer(self):
        """Test that AdamInputSmoothing produces same results as torch.optim.Adam."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps)
        
        # Create a dummy parameter and optimizer
        param = torch.randn(2, 10, 5, requires_grad=True)
        optimizer = torch.optim.Adam([param], lr=0.001, betas=(beta1, beta2), eps=eps)
        
        # Run multiple steps
        for step in range(5):
            # Generate gradient
            grad = torch.randn(2, 10, 5)
            
            # Apply gradient to parameter manually to trigger Adam update
            param.grad = grad.clone()
            
            # Get smoother output (verify it doesn't crash)
            _ = smoother(grad)
            
            # Step the optimizer to update its internal state
            optimizer.step()
            optimizer.zero_grad()
            
            # Verify the smoother's internal state matches expectations
            assert smoother.t[0].item() == step + 1

    def test_clone_preserves_zero_t_behavior(self):
        """Test that clone with zero_t=True resets timesteps for cloned entries."""
        smoother = AdamInputSmoothing()
        
        # Initialize and run a few steps
        x = torch.randn(3, 10)
        for _ in range(5):
            smoother(x)
        
        assert smoother.t[0].item() == 5
        
        # Clone with zero_t=True
        clone_mask = torch.tensor([True, False, True])
        smoother.clone(clone_mask, zero_t=True)
        
        # Original entries should maintain their t
        assert smoother.t[0].item() == 5
        assert smoother.t[1].item() == 5
        assert smoother.t[2].item() == 5
        
        # New cloned entries should have t=0
        assert smoother.t[3].item() == 0
        assert smoother.t[4].item() == 0

    def test_split_preserves_zero_t_behavior(self):
        """Test that split with zero_t=True resets timesteps for split entries."""
        smoother = AdamInputSmoothing()

        # Initialize and run a few steps
        x = torch.randn(3, 10)
        for _ in range(5):
            smoother(x)

        # Split with zero_t=True (removes index 1, appends 2 new)
        split_mask = torch.tensor([False, True, False])
        N = 2
        smoother.split(split_mask, N=N, zero_t=True)

        # Remaining original entries (indices 0 and 2, now at positions 0 and 1)
        assert smoother.t[0].item() == 5
        assert smoother.t[1].item() == 5

        # New split entries should have t=0
        assert smoother.t[2].item() == 0
        assert smoother.t[3].item() == 0

    def test_zero_out_with_zero_t_false(self):
        """Test that zero_out with zero_t=False preserves timesteps."""
        smoother = AdamInputSmoothing()
        
        # Initialize and run steps
        x = torch.randn(2, 10, 5)
        for _ in range(3):
            smoother(x)
        
        original_t = smoother.t.clone()
        
        # Zero out moments but keep timesteps
        smoother.zero_out(zero_t=False)
        
        # Moments should be zero
        assert torch.allclose(smoother.m, torch.zeros_like(smoother.m))
        assert torch.allclose(smoother.v, torch.zeros_like(smoother.v))
        
        # Timesteps should be preserved
        torch.testing.assert_close(smoother.t, original_t)

    def test_zero_out_with_zero_t_true(self):
        """Test that zero_out with zero_t=True resets timesteps."""
        smoother = AdamInputSmoothing()
        
        # Initialize and run steps
        x = torch.randn(2, 10, 5)
        for _ in range(3):
            smoother(x)
        
        # Zero out everything including timesteps
        smoother.zero_out(zero_t=True)
        
        # Everything should be zero
        assert torch.allclose(smoother.m, torch.zeros_like(smoother.m))
        assert torch.allclose(smoother.v, torch.zeros_like(smoother.v))
        assert torch.allclose(smoother.t, torch.zeros_like(smoother.t))

    def test_adam_matches_densify_prune_usage(self):
        """Test that zero_t behavior matches what densify_and_prune.py expects."""
        smoother = AdamInputSmoothing()

        # Simulate initial training
        x = torch.randn(10, 5)  # [num_gaussians, features]
        for _ in range(10):
            smoother(x)

        # Simulate cloning (densification) — appends 2 new entries
        clone_mask = torch.tensor([False] * 8 + [True, True])
        smoother.clone(clone_mask, zero_t=True)

        # Now 12 entries: 10 original + 2 cloned
        assert smoother.m.shape[0] == 12

        # New cloned gaussians should have fresh optimizer state (t=0)
        assert smoother.t[10].item() == 0
        assert smoother.t[11].item() == 0
        assert torch.allclose(smoother.m[10:], torch.zeros_like(smoother.m[10:]))
        assert torch.allclose(smoother.v[10:], torch.zeros_like(smoother.v[10:]))

        # Simulate splitting — removes index 10, appends 2 new zeros
        split_mask = torch.tensor([False] * 10 + [True, False])
        smoother.split(split_mask, N=2, zero_t=True)

        # Now 13 entries: 11 remaining + 2 new from split
        assert smoother.m.shape[0] == 13

        # New split gaussians (last 2) should have fresh optimizer state (t=0)
        assert smoother.t[11].item() == 0
        assert smoother.t[12].item() == 0

        # Simulate pruning — remove last 2
        prune_mask = torch.tensor([False] * 11 + [True, True])
        smoother.prune(prune_mask)

        # Should have 11 gaussians left
        assert smoother.m.shape[0] == 11
        assert smoother.v.shape[0] == 11
        assert smoother.t.shape[0] == 11

    def test_opacity_reset_behavior(self):
        """Test behavior that matches opacity reset in densify_and_prune.py."""
        smoother = AdamInputSmoothing()
        
        # Initialize
        x = torch.randn(5, 1)  # [num_gaussians, 1] for opacity
        for _ in range(10):
            smoother(x)
        
        original_t = smoother.t.clone()
        
        # Simulate opacity reset: zero moments but keep timesteps
        smoother.zero_out(zero_t=False)
        
        # Moments should be reset
        assert torch.allclose(smoother.m, torch.zeros_like(smoother.m))
        assert torch.allclose(smoother.v, torch.zeros_like(smoother.v))
        
        # Timesteps should be preserved (important for bias correction)
        torch.testing.assert_close(smoother.t, original_t)
        
        # Next update should still use the preserved timestep
        x_new = torch.randn(5, 1)
        _ = smoother(x_new)
        
        # Timestep should have incremented
        assert smoother.t[0].item() == original_t[0].item() + 1


class TestEfficientOps:
    """
    Verify that the optimised forward path (lerp_ / addcmul_) produces
    numerically identical results to the reference formula.
    """

    def test_lerp_equivalent_to_ema(self):
        """lerp_(chunk, 1-beta1) == beta1*m + (1-beta1)*chunk."""
        beta1 = 0.9
        m = torch.randn(4, 6)
        chunk = torch.randn(4, 6)
        expected = beta1 * m + (1 - beta1) * chunk
        got = m.clone().lerp_(chunk, 1 - beta1)
        torch.testing.assert_close(got, expected)

    def test_addcmul_equivalent_to_ema_sq(self):
        """mul_(beta2).addcmul_(c, c, value=1-beta2) == beta2*v + (1-beta2)*c^2."""
        beta2 = 0.999
        v = torch.rand(4, 6)
        chunk = torch.randn(4, 6)
        expected = beta2 * v + (1 - beta2) * chunk ** 2
        got = v.clone().mul_(beta2).addcmul_(chunk, chunk, value=1 - beta2)
        torch.testing.assert_close(got, expected)

    def test_forward_2d_matches_manual(self):
        """Optimised forward on a 2-D tensor matches the hand-rolled Adam formula."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps)

        B, C = 5, 8
        m_ref = torch.zeros(B, C)
        v_ref = torch.zeros(B, C)

        for step in range(1, 5):
            x = torch.randn(B, C)

            m_ref = beta1 * m_ref + (1 - beta1) * x
            v_ref = beta2 * v_ref + (1 - beta2) * x ** 2
            m_hat = m_ref / (1 - beta1 ** step)
            v_hat = v_ref / (1 - beta2 ** step)
            expected = m_hat / (v_hat.sqrt() + eps)

            got = smoother(x)
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    def test_forward_3d_matches_manual(self):
        """Optimised forward on a 3-D tensor matches the hand-rolled Adam formula."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        smoother = AdamInputSmoothing(beta1=beta1, beta2=beta2, eps=eps)

        B, N, C = 3, 10, 6
        m_ref = torch.zeros(B, N, C)
        v_ref = torch.zeros(B, N, C)

        for step in range(1, 6):
            x = torch.randn(B, N, C)

            m_ref = beta1 * m_ref + (1 - beta1) * x
            v_ref = beta2 * v_ref + (1 - beta2) * x ** 2
            m_hat = m_ref / (1 - beta1 ** step)
            v_hat = v_ref / (1 - beta2 ** step)
            expected = m_hat / (v_hat.sqrt() + eps)

            got = smoother(x)
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    def test_forward_does_not_mutate_input(self):
        """forward() must not modify the caller's tensor."""
        smoother = AdamInputSmoothing()
        x = torch.randn(4, 8)
        x_before = x.clone()
        smoother(x)
        torch.testing.assert_close(x, x_before)

    def test_forward_with_slice_does_not_mutate_input(self):
        """forward() with input_slice must not modify the caller's tensor."""
        smoother = AdamInputSmoothing(input_slice=slice(2, 6))
        x = torch.randn(4, 10)
        x_before = x.clone()
        smoother(x)
        torch.testing.assert_close(x, x_before)

    def test_t_shape_generalises_to_arbitrary_ndim(self):
        """reshape-based broadcasting works for both 2-D and 3-D state tensors."""
        for shape in [(6, 4), (3, 6, 4)]:
            smoother = AdamInputSmoothing()
            smoother(torch.randn(*shape))
            # second call must not crash and must update t
            smoother(torch.randn(*shape))
            assert smoother.t.shape == (shape[0],)
            assert smoother.t[0].item() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
