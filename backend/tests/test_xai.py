"""
XAI Integrity Tests

These tests verify the gradient-based input saliency implementation
for scientific correctness and auditability.

Run with: python -m pytest backend/tests/test_xai.py -v
"""

import torch
import numpy as np
import pytest
import tempfile
import os

from app.services.xai import (
    generate_gradient_saliency,
    compute_segmentation_alignment,
    XAIProvenance,
)
from ai_pipeline.models.armt_gan import ARMTGenerator2D


class TestXAIProvenance:
    """Test XAIProvenance dataclass."""

    def test_provenance_defaults(self):
        """Test default provenance values."""
        prov = XAIProvenance()
        assert prov.method == "gradient-based input saliency"
        assert prov.target == "sum of tumor logits across spatial dimensions"
        assert prov.normalization == "min-max [0, 1]"
        assert prov.aggregation == "max absolute gradient across input channels"
        assert prov.model_checkpoint is None
        assert prov.model_version is None
        assert prov.input_shape is None
        assert prov.output_shape is None

    def test_provenance_with_values(self):
        """Test provenance with custom values."""
        prov = XAIProvenance(
            model_checkpoint="generator_latest.pth",
            model_version="armt-gan-2d-baseline",
            input_shape=(1, 4, 224, 224),
            output_shape=(1, 1, 224, 224),
        )
        assert prov.model_checkpoint == "generator_latest.pth"
        assert prov.model_version == "armt-gan-2d-baseline"
        assert prov.input_shape == (1, 4, 224, 224)
        assert prov.output_shape == (1, 1, 224, 224)

    def test_provenance_to_dict(self):
        """Test provenance serialization."""
        prov = XAIProvenance(
            model_checkpoint="test.pth",
            model_version="v1",
            input_shape=(1, 4, 224, 224),
            output_shape=(1, 1, 224, 224),
        )
        d = prov.to_dict()
        assert d["method"] == "gradient-based input saliency"
        assert d["target"] == "sum of tumor logits across spatial dimensions"
        assert d["normalization"] == "min-max [0, 1]"
        assert d["model_checkpoint"] == "test.pth"
        assert d["model_version"] == "v1"
        assert d["input_shape"] == [1, 4, 224, 224]
        assert d["output_shape"] == [1, 1, 224, 224]


class TestGenerateGradientSaliency:
    """Test gradient-based input saliency generation."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        model = ARMTGenerator2D(in_channels=4, out_channels=1, features=[16, 32])
        model.eval()
        return model

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor."""
        torch.manual_seed(42)
        return torch.randn(1, 4, 224, 224, requires_grad=False)

    def test_output_shape(self, model, input_tensor):
        """Test that output shape matches expected [B, 1, H, W]."""
        saliency, provenance = generate_gradient_saliency(input_tensor, model)
        
        assert saliency.shape == (1, 1, 224, 224)
        assert provenance.output_shape == (1, 1, 224, 224)
        assert provenance.input_shape == (1, 4, 224, 224)

    def test_output_range(self, model, input_tensor):
        """Test that output values are in [0, 1]."""
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        
        assert saliency.min() >= 0.0
        assert saliency.max() <= 1.0

    def test_normalization_non_flat(self, model, input_tensor):
        """Test normalization with non-flat gradients."""
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        
        # Should have variation (not all same value)
        assert saliency.max() > saliency.min()

    def test_deterministic(self, model, input_tensor):
        """Test that repeated calls produce identical results."""
        saliency1, _ = generate_gradient_saliency(input_tensor, model)
        saliency2, _ = generate_gradient_saliency(input_tensor, model)
        
        np.testing.assert_array_equal(saliency1, saliency2)

    def test_no_weight_modification(self, model, input_tensor):
        """Test that XAI does not modify model weights."""
        # Capture initial weights
        initial_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        generate_gradient_saliency(input_tensor, model)
        
        # Check weights unchanged
        for name, param in model.named_parameters():
            assert torch.equal(param, initial_weights[name]), f"Weight {name} was modified!"

    def test_model_in_eval_mode(self, model, input_tensor):
        """Test that model is set to eval mode."""
        # Set model to train mode first
        model.train()
        assert model.training
        
        generate_gradient_saliency(input_tensor, model)
        
        # Should be in eval mode after call
        assert not model.training

    def test_gradient_availability(self, model, input_tensor):
        """Test that gradients are properly computed."""
        # This should not raise an error
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        assert saliency is not None

    def test_input_shape_validation(self, model):
        """Test that wrong input shape raises ValueError."""
        # Wrong number of channels
        bad_input = torch.randn(1, 3, 224, 224)
        with pytest.raises(ValueError, match="Expected input shape"):
            generate_gradient_saliency(bad_input, model)
        
        # Wrong number of dimensions
        bad_input = torch.randn(4, 224, 224)
        with pytest.raises(ValueError, match="Expected input shape"):
            generate_gradient_saliency(bad_input, model)

    def test_provenance_fields_populated(self, model, input_tensor):
        """Test that provenance contains all required fields."""
        _, provenance = generate_gradient_saliency(
            input_tensor, 
            model,
            model_checkpoint="test.pth",
            model_version="test-v1",
        )
        
        assert provenance.method == "gradient-based input saliency"
        assert provenance.target == "sum of tumor logits across spatial dimensions"
        assert provenance.normalization == "min-max [0, 1]"
        assert provenance.aggregation == "max absolute gradient across input channels"
        assert provenance.model_checkpoint == "test.pth"
        assert provenance.model_version == "test-v1"
        assert provenance.input_shape == (1, 4, 224, 224)
        assert provenance.output_shape == (1, 1, 224, 224)

    def test_batch_size_greater_than_one(self, model):
        """Test with batch size > 1."""
        torch.manual_seed(123)
        input_tensor = torch.randn(2, 4, 112, 112)
        
        saliency, provenance = generate_gradient_saliency(input_tensor, model)
        
        assert saliency.shape == (2, 1, 112, 112)
        assert provenance.input_shape == (2, 4, 112, 112)
        assert provenance.output_shape == (2, 1, 112, 112)

    def test_different_spatial_sizes(self, model):
        """Test with different spatial dimensions."""
        for h, w in [(112, 112), (224, 224), (256, 256), (128, 128)]:
            torch.manual_seed(42)
            input_tensor = torch.randn(1, 4, h, w)
            saliency, provenance = generate_gradient_saliency(input_tensor, model)
            
            assert saliency.shape == (1, 1, h, w)
            assert provenance.output_shape == (1, 1, h, w)


class TestFlatGradientHandling:
    """Test handling of flat (constant) gradients."""

    def test_constant_output_model(self):
        """Test with a model that produces constant output."""
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                # Use x in a way that produces constant output but preserves grad flow
                return torch.zeros_like(x[:, :1, :, :]) + (x * 0).sum()
        
        model = ConstantModel()
        model.eval()
        input_tensor = torch.randn(1, 4, 224, 224)
        
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        
        # Should return zeros without error
        assert saliency.shape == (1, 1, 224, 224)
        assert np.allclose(saliency, 0.0)

    def test_constant_nonzero_output_model(self):
        """Test with a model that produces constant non-zero output."""
        class ConstantNonzeroModel(torch.nn.Module):
            def forward(self, x):
                # Use x in a way that produces constant output but preserves grad flow
                return torch.ones_like(x[:, :1, :, :]) * 5.0 + (x * 0).sum()
        
        model = ConstantNonzeroModel()
        model.eval()
        input_tensor = torch.randn(1, 4, 224, 224)
        
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        
        # Should return zeros (flat gradient -> zero attribution)
        assert saliency.shape == (1, 1, 224, 224)
        assert np.allclose(saliency, 0.0)


class TestGradientUnavailableError:
    """Test error handling when gradients are unavailable."""

    def test_non_differentiable_model(self):
        """Test error when model breaks gradient flow."""
        class NonDifferentiableModel(torch.nn.Module):
            def forward(self, x):
                # Detach breaks gradient flow
                return x[:, :1, :, :].detach()
        
        model = NonDifferentiableModel()
        model.eval()
        input_tensor = torch.randn(1, 4, 224, 224)
        
        with pytest.raises(RuntimeError, match="does not require grad|Gradients unavailable"):
            generate_gradient_saliency(input_tensor, model)

    def test_no_grad_context(self):
        """Test error when called in no_grad context."""
        model = ARMTGenerator2D(in_channels=4, out_channels=1, features=[16, 32])
        model.eval()
        input_tensor = torch.randn(1, 4, 224, 224)
        
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="does not require grad|Gradients unavailable"):
                generate_gradient_saliency(input_tensor, model)


class TestSegmentationAlignment:
    """Test segmentation alignment metric."""

    def test_perfect_alignment(self):
        """Test alignment when attribution matches segmentation."""
        attribution = np.zeros((1, 1, 10, 10))
        attribution[0, 0, 3:7, 3:7] = 1.0  # Square in center
        
        mask = np.zeros((1, 1, 10, 10))
        mask[0, 0, 3:7, 3:7] = 1.0  # Same square
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        assert result["mean_inside"] == 1.0
        assert result["mean_outside"] == 0.0
        assert result["ratio"] is None  # inf case
        assert result["tumor_pixel_count"] == 16
        assert result["background_pixel_count"] == 84

    def test_no_overlap(self):
        """Test alignment when attribution is outside segmentation."""
        attribution = np.zeros((1, 1, 10, 10))
        attribution[0, 0, 0:4, 0:4] = 1.0  # Top-left (16 pixels)
        
        mask = np.zeros((1, 1, 10, 10))
        mask[0, 0, 6:10, 6:10] = 1.0  # Bottom-right (16 pixels)
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        assert result["mean_inside"] == 0.0
        # Outside region: 84 pixels, 16 have value 1.0, 68 have value 0.0
        assert abs(result["mean_outside"] - (16.0 / 84)) < 1e-6
        assert result["ratio"] == 0.0
        assert result["tumor_pixel_count"] == 16
        assert result["background_pixel_count"] == 84

    def test_partial_overlap(self):
        """Test alignment with partial overlap."""
        attribution = np.ones((1, 1, 10, 10)) * 0.5
        attribution[0, 0, 3:7, 3:7] = 1.0  # Higher in center
        
        mask = np.zeros((1, 1, 10, 10))
        mask[0, 0, 3:7, 3:7] = 1.0  # Center square
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        assert result["mean_inside"] == 1.0
        assert result["mean_outside"] == 0.5
        assert result["ratio"] == 2.0
        assert result["tumor_pixel_count"] == 16

    def test_empty_tumor_mask(self):
        """Test alignment with empty tumor mask."""
        attribution = np.ones((1, 1, 10, 10)) * 0.5
        mask = np.zeros((1, 1, 10, 10))  # No tumor
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        assert result["mean_inside"] == 0.0
        assert result["mean_outside"] == 0.5
        assert result["ratio"] == 0.0
        assert result["tumor_pixel_count"] == 0
        assert result["background_pixel_count"] == 100
        assert "note" in result

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        attribution = np.zeros((1, 1, 10, 10))
        mask = np.zeros((1, 1, 8, 8))
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_segmentation_alignment(attribution, mask)

    def test_squeezed_inputs(self):
        """Test with already-squeezed [H, W] inputs."""
        attribution = np.zeros((10, 10))
        attribution[3:7, 3:7] = 1.0
        
        mask = np.zeros((10, 10))
        mask[3:7, 3:7] = 1.0
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        assert result["mean_inside"] == 1.0
        assert result["mean_outside"] == 0.0

    def test_probability_mask_threshold(self):
        """Test with probability mask (not binary) using threshold."""
        attribution = np.ones((1, 1, 10, 10)) * 0.5
        attribution[0, 0, 3:7, 3:7] = 1.0
        
        # Probability mask with values between 0 and 1
        mask = np.zeros((1, 1, 10, 10))
        mask[0, 0, 3:7, 3:7] = 0.8  # High probability
        mask[0, 0, 7:9, 7:9] = 0.3  # Low probability (below 0.5 threshold)
        
        result = compute_segmentation_alignment(attribution, mask, threshold=0.5)
        
        # Only the 0.8 region should be considered tumor
        assert result["tumor_pixel_count"] == 16
        assert result["mean_inside"] == 1.0


class TestBackwardCompatibility:
    """Test backward compatibility alias."""

    def test_generate_gradcam_alias(self):
        """Test that generate_gradcam alias still works with warning."""
        from app.services.xai import generate_gradcam
        model = ARMTGenerator2D(in_channels=4, out_channels=1, features=[16, 32])
        model.eval()
        input_tensor = torch.randn(1, 4, 224, 224)
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            saliency = generate_gradcam(input_tensor, model)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "generate_gradcam is deprecated" in str(w[0].message)
        
        assert saliency.shape == (1, 1, 224, 224)
        assert saliency.min() >= 0.0
        assert saliency.max() <= 1.0


class TestIntegrationWithPipeline:
    """Integration tests with the ARMT-GAN model."""

    def test_full_pipeline_deterministic(self):
        """Test full XAI pipeline is deterministic."""
        model = ARMTGenerator2D(in_channels=4, out_channels=1, features=[16, 32, 64])
        model.eval()
        
        torch.manual_seed(42)
        input1 = torch.randn(1, 4, 224, 224)
        saliency1, prov1 = generate_gradient_saliency(input1, model)
        
        torch.manual_seed(42)
        input2 = torch.randn(1, 4, 224, 224)
        saliency2, prov2 = generate_gradient_saliency(input2, model)
        
        np.testing.assert_array_equal(saliency1, saliency2)
        assert prov1.to_dict() == prov2.to_dict()

    def test_xai_then_inference_no_interference(self):
        """Test that XAI computation doesn't affect subsequent inference."""
        model = ARMTGenerator2D(in_channels=4, out_channels=1, features=[16, 32, 64])
        model.eval()
        
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 4, 224, 224)
        
        # Run XAI
        saliency, _ = generate_gradient_saliency(input_tensor, model)
        
        # Run inference
        with torch.inference_mode():
            mask = model(input_tensor)
        
        # Both should succeed
        assert saliency.shape == (1, 1, 224, 224)
        assert mask.shape == (1, 1, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])