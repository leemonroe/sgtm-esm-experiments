import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import esm

from sgtm.masking import build_sgtm_masks, adjust_gradients, ablate, register_gradient_routing_hooks


NUM_LAYERS = 6
FORGET_HEADS = [17, 18, 19]
HEAD_DIM = 16
FORGET_ROWS_START = 272
FORGET_ROWS_END = 320
FORGET_MLP_START = 1120

MASKED_SUBMODULES = [
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.out_proj.weight",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
]


@pytest.fixture(scope="session")
def model_and_masks():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    retain_mask, forget_mask = build_sgtm_masks(model)
    return model, alphabet, retain_mask, forget_mask


class TestBuildMasks:
    def test_mask_shapes(self, model_and_masks):
        model, _, retain_mask, forget_mask = model_and_masks
        param_dict = dict(model.named_parameters())
        for name in forget_mask:
            assert forget_mask[name].shape == param_dict[name].shape
            assert retain_mask[name].shape == param_dict[name].shape

    def test_masks_complementary(self, model_and_masks):
        _, _, retain_mask, forget_mask = model_and_masks
        for name in forget_mask:
            combined = retain_mask[name] | forget_mask[name]
            assert combined.all(), f"OR not all True for {name}"
            overlap = retain_mask[name] & forget_mask[name]
            assert not overlap.any(), f"AND not all False for {name}"

    def test_correct_params_masked(self, model_and_masks):
        _, _, _, forget_mask = model_and_masks
        expected_names = set()
        for layer_idx in range(NUM_LAYERS):
            for sub in MASKED_SUBMODULES:
                expected_names.add(f"layers.{layer_idx}.{sub}")
        assert set(forget_mask.keys()) == expected_names

    def test_shared_biases_excluded(self, model_and_masks):
        _, _, _, forget_mask = model_and_masks
        for layer_idx in range(NUM_LAYERS):
            assert f"layers.{layer_idx}.self_attn.out_proj.bias" not in forget_mask
            assert f"layers.{layer_idx}.fc2.bias" not in forget_mask

    def test_masks_no_grad(self, model_and_masks):
        _, _, retain_mask, forget_mask = model_and_masks
        for name in forget_mask:
            assert not forget_mask[name].requires_grad
            assert not retain_mask[name].requires_grad


class TestAdjustGradients:
    def _forward_backward(self, model, alphabet):
        batch_converter = alphabet.get_batch_converter()
        data = [("test", "MKTLLILAVL")]
        _, _, tokens = batch_converter(data)
        model.train()
        out = model(tokens, repr_layers=[6])
        loss = out["logits"].sum()
        loss.backward()

    def test_forget_mode(self, model_and_masks):
        model, alphabet, retain_mask, forget_mask = model_and_masks
        self._forward_backward(model, alphabet)
        adjust_gradients(model, retain_mask, forget_mask, "forget")

        for name, param in model.named_parameters():
            if name not in forget_mask or param.grad is None:
                continue
            retain_grad = param.grad[retain_mask[name]]
            assert (retain_grad == 0).all(), f"Retain region not zeroed for {name}"
            forget_grad = param.grad[forget_mask[name]]
            assert forget_grad.abs().sum() > 0, f"Forget region unexpectedly zero for {name}"

        model.zero_grad()

    def test_retain_mode(self, model_and_masks):
        model, alphabet, retain_mask, forget_mask = model_and_masks
        self._forward_backward(model, alphabet)
        adjust_gradients(model, retain_mask, forget_mask, "retain")

        for name, param in model.named_parameters():
            if name not in forget_mask or param.grad is None:
                continue
            forget_grad = param.grad[forget_mask[name]]
            assert (forget_grad == 0).all(), f"Forget region not zeroed for {name}"
            retain_grad = param.grad[retain_mask[name]]
            assert retain_grad.abs().sum() > 0, f"Retain region unexpectedly zero for {name}"

        model.zero_grad()

    def test_default_mode(self, model_and_masks):
        model, alphabet, retain_mask, forget_mask = model_and_masks
        self._forward_backward(model, alphabet)

        grads_before = {}
        for name, param in model.named_parameters():
            if name in forget_mask and param.grad is not None:
                grads_before[name] = param.grad.clone()

        adjust_gradients(model, retain_mask, forget_mask, "default")

        for name, param in model.named_parameters():
            if name in grads_before:
                assert torch.equal(param.grad, grads_before[name]), \
                    f"Grad changed in default mode for {name}"

        model.zero_grad()


class TestAblate:
    def test_ablation(self, model_and_masks):
        _, alphabet, _, forget_mask = model_and_masks
        model = copy.deepcopy(model_and_masks[0])

        retain_mask, _ = build_sgtm_masks(model)
        retain_values = {}
        for name, param in model.named_parameters():
            if name in forget_mask:
                retain_values[name] = param.data[retain_mask[name]].clone()

        ablate(model, forget_mask)

        for name, param in model.named_parameters():
            if name in forget_mask:
                forget_vals = param.data[forget_mask[name]]
                assert (forget_vals == 0).all(), f"Forget params not zeroed for {name}"

                retained = param.data[retain_mask[name]]
                assert torch.equal(retained, retain_values[name]), \
                    f"Retain params changed for {name}"

    def test_model_valid_after_ablation(self, model_and_masks):
        _, alphabet, _, forget_mask = model_and_masks
        model = copy.deepcopy(model_and_masks[0])
        ablate(model, forget_mask)

        model.eval()
        batch_converter = alphabet.get_batch_converter()
        data = [("test", "MKTLLILAVL")]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            out = model(tokens)

        assert not torch.isnan(out["logits"]).any()


class TestGradientRouting:
    def _forward_backward(self, model, alphabet):
        batch_converter = alphabet.get_batch_converter()
        data = [("test", "MKTLLILAVL")]
        _, _, tokens = batch_converter(data)
        model.train()
        out = model(tokens, repr_layers=[6])
        loss = out["logits"].sum()
        loss.backward()

    def test_hooks_register(self, model_and_masks):
        model = copy.deepcopy(model_and_masks[0])
        handles, mode_ref = register_gradient_routing_hooks(model)
        assert len(handles) == 12  # 6 layers × 2 hooks (out_proj + fc2)
        assert mode_ref == ["default"]
        for h in handles:
            h.remove()

    def test_forget_mode_routes_gradients(self, model_and_masks):
        """In forget mode, retain params in fc1/q/k/v should get ~zero gradients."""
        model = copy.deepcopy(model_and_masks[0])
        _, alphabet, retain_mask, forget_mask = model_and_masks
        handles, mode_ref = register_gradient_routing_hooks(model)

        mode_ref[0] = "forget"
        self._forward_backward(model, alphabet)

        # fc1 retain params should have zero gradients (gradient flow cut by detach on fc2 input)
        for name, param in model.named_parameters():
            if name not in forget_mask or param.grad is None:
                continue
            if "fc1" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
                retain_grad = param.grad[retain_mask[name]]
                assert (retain_grad == 0).all(), f"Retain grads not zero for {name}"

        model.zero_grad()
        for h in handles:
            h.remove()

    def test_retain_mode_routes_gradients(self, model_and_masks):
        """In retain mode, forget params in fc1/q/k/v should get ~zero gradients."""
        model = copy.deepcopy(model_and_masks[0])
        _, alphabet, retain_mask, forget_mask = model_and_masks
        handles, mode_ref = register_gradient_routing_hooks(model)

        mode_ref[0] = "retain"
        self._forward_backward(model, alphabet)

        # fc1 forget params should have zero gradients
        for name, param in model.named_parameters():
            if name not in forget_mask or param.grad is None:
                continue
            if "fc1" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
                forget_grad = param.grad[forget_mask[name]]
                assert (forget_grad == 0).all(), f"Forget grads not zero for {name}"

        model.zero_grad()
        for h in handles:
            h.remove()

    def test_default_mode_no_routing(self, model_and_masks):
        """In default mode, all params should get non-zero gradients."""
        model = copy.deepcopy(model_and_masks[0])
        _, alphabet, retain_mask, forget_mask = model_and_masks
        handles, mode_ref = register_gradient_routing_hooks(model)

        mode_ref[0] = "default"
        self._forward_backward(model, alphabet)

        for name, param in model.named_parameters():
            if name not in forget_mask or param.grad is None:
                continue
            assert param.grad.abs().sum() > 0, f"All grads zero for {name}"

        model.zero_grad()
        for h in handles:
            h.remove()

    def test_output_valid_with_hooks(self, model_and_masks):
        """Model should produce valid (non-NaN) output with hooks active."""
        model = copy.deepcopy(model_and_masks[0])
        _, alphabet, _, _ = model_and_masks
        handles, mode_ref = register_gradient_routing_hooks(model)

        batch_converter = alphabet.get_batch_converter()
        data = [("test", "MKTLLILAVL")]
        _, _, tokens = batch_converter(data)

        for mode in ("default", "forget", "retain"):
            mode_ref[0] = mode
            model.eval()
            with torch.no_grad():
                out = model(tokens)
            assert not torch.isnan(out["logits"]).any(), f"NaN in {mode} mode"

        for h in handles:
            h.remove()
