"""
Shared model configuration for ESM-2 SGTM experiments.

Single source of truth for architecture dimensions, forget parameter defaults,
and alphabet loading. Prevents hardcoded 8M values scattered across scripts.
"""

from dataclasses import dataclass, field
from typing import List

import esm


@dataclass(frozen=True)
class ESM2Config:
    name: str
    pretrained_fn: str
    num_layers: int
    embed_dim: int
    attention_heads: int
    head_dim: int
    mlp_dim: int
    default_forget_heads: List[int]
    default_forget_mlp_start: int


# Registry of supported model sizes
MODEL_CONFIGS = {
    "8M": ESM2Config(
        name="8M",
        pretrained_fn="esm2_t6_8M_UR50D",
        num_layers=6,
        embed_dim=320,
        attention_heads=20,
        head_dim=16,
        mlp_dim=1280,
        default_forget_heads=[17, 18, 19],
        default_forget_mlp_start=1120,
    ),
    "35M": ESM2Config(
        name="35M",
        pretrained_fn="esm2_t12_35M_UR50D",
        num_layers=12,
        embed_dim=480,
        attention_heads=20,
        head_dim=24,
        mlp_dim=1920,
        default_forget_heads=[18, 19],
        default_forget_mlp_start=1680,
    ),
}


def get_config(model_size: str) -> ESM2Config:
    """Get config by model size string (e.g. '8M', '35M')."""
    key = model_size.upper()
    if key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[key]


def load_alphabet() -> esm.data.Alphabet:
    """Load ESM alphabet without downloading pretrained weights."""
    return esm.data.Alphabet.from_architecture("ESM-1b")


def create_model(config: ESM2Config, alphabet=None):
    """Create a fresh (randomly initialized) ESM-2 model from config."""
    if alphabet is None:
        alphabet = load_alphabet()
    model = esm.model.esm2.ESM2(
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        attention_heads=config.attention_heads,
        alphabet=alphabet,
    )
    return model, alphabet


def load_model_from_checkpoint(config: ESM2Config, checkpoint_path: str, device: str):
    """Load a trained ESM-2 model from a state_dict checkpoint."""
    import torch
    model, alphabet = create_model(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, alphabet
