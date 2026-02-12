import torch
from typing import Dict, List, Tuple


def build_sgtm_masks(
    model,
    forget_head_indices: List[int] = [17, 18, 19],
    forget_mlp_start: int = 1120,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    head_dim = 16
    forget_rows_start = min(forget_head_indices) * head_dim  # 272
    forget_rows_end = (max(forget_head_indices) + 1) * head_dim  # 320

    retain_mask = {}
    forget_mask = {}

    for name, param in model.named_parameters():
        if not name.startswith("layers."):
            continue

        parts = name.split(".")
        layer_idx = int(parts[1])
        submodule = ".".join(parts[2:])

        mask = torch.zeros_like(param, dtype=torch.bool)

        if submodule in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight"):
            mask[forget_rows_start:forget_rows_end, :] = True

        elif submodule in ("self_attn.q_proj.bias", "self_attn.k_proj.bias",
                           "self_attn.v_proj.bias"):
            mask[forget_rows_start:forget_rows_end] = True

        elif submodule == "self_attn.out_proj.weight":
            mask[:, forget_rows_start:forget_rows_end] = True

        elif submodule == "fc1.weight":
            mask[forget_mlp_start:, :] = True

        elif submodule == "fc1.bias":
            mask[forget_mlp_start:] = True

        elif submodule == "fc2.weight":
            mask[:, forget_mlp_start:] = True

        else:
            # Shared biases (out_proj.bias, fc2.bias) and layer norms: skip
            continue

        forget_mask[name] = mask
        retain_mask[name] = ~mask

    return retain_mask, forget_mask


def adjust_gradients(
    model,
    retain_mask: Dict[str, torch.Tensor],
    forget_mask: Dict[str, torch.Tensor],
    sgtm_mode: str,
) -> None:
    if sgtm_mode == "default":
        return

    for name, param in model.named_parameters():
        if name not in forget_mask:
            continue
        if param.grad is None:
            continue

        if sgtm_mode == "forget":
            param.grad[retain_mask[name]] = 0.0
        elif sgtm_mode == "retain":
            param.grad[forget_mask[name]] = 0.0
        else:
            raise ValueError(f"Unknown sgtm_mode: {sgtm_mode}")


def register_gradient_routing_hooks(
    model,
    forget_head_indices: List[int] = [17, 18, 19],
    forget_mlp_start: int = 1120,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], List]:
    """Register forward pre-hooks that detach activation dims to route gradients.

    Instead of zeroing gradients post-backward, this detaches activation dims
    during the forward pass so gradients never flow through them:
    - "forget" mode: detach retain dims (only forget params get gradients)
    - "retain" mode: detach forget dims (only retain params get gradients)
    - "default" mode: no detaching (all params get gradients)

    Returns (hook_handles, sgtm_mode_ref) where sgtm_mode_ref is a mutable
    list ["default"] that the training loop sets before each forward pass.
    """
    head_dim = 16
    forget_start = min(forget_head_indices) * head_dim
    forget_end = (max(forget_head_indices) + 1) * head_dim

    sgtm_mode_ref = ["default"]
    handles = []

    def _detach_dims(x, retain_slice, forget_slice, mode):
        """Detach a slice of dimensions from tensor x, returning modified x."""
        if mode == "default":
            return x
        out = x.clone()
        if mode == "forget":
            # Detach retain dims so only forget params get gradients
            out[..., retain_slice] = x[..., retain_slice].detach()
        elif mode == "retain":
            # Detach forget dims so only retain params get gradients
            out[..., forget_slice] = x[..., forget_slice].detach()
        return out

    # Slices for attention (out_proj input) and MLP (fc2 input)
    attn_retain_slice = slice(0, forget_start)
    attn_forget_slice = slice(forget_start, forget_end)
    mlp_retain_slice = slice(0, forget_mlp_start)
    mlp_forget_slice = slice(forget_mlp_start, None)

    for layer in model.layers:
        # Pre-hook on out_proj: intercepts merged attention head output
        def make_attn_hook():
            def hook(module, args):
                mode = sgtm_mode_ref[0]
                if mode == "default":
                    return args
                x = args[0]
                x = _detach_dims(x, attn_retain_slice, attn_forget_slice, mode)
                return (x,) + args[1:]
            return hook

        h = layer.self_attn.out_proj.register_forward_pre_hook(make_attn_hook())
        handles.append(h)

        # Pre-hook on fc2: intercepts GELU(fc1(x)) output
        def make_mlp_hook():
            def hook(module, args):
                mode = sgtm_mode_ref[0]
                if mode == "default":
                    return args
                x = args[0]
                x = _detach_dims(x, mlp_retain_slice, mlp_forget_slice, mode)
                return (x,) + args[1:]
            return hook

        h = layer.fc2.register_forward_pre_hook(make_mlp_hook())
        handles.append(h)

    return handles, sgtm_mode_ref


def ablate(model, forget_mask: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in forget_mask:
                param[forget_mask[name]] = 0.0
