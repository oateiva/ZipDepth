"""torch.compile wrapper that degrades gracefully and works around Windows issues."""

import importlib.util
import sys

import torch
import torch.nn as nn


def compile_model(model: nn.Module, mode: str = "reduce-overhead", **kwargs) -> nn.Module:
    """Return ``torch.compile(model)``, or ``model`` unchanged if Triton is unavailable."""
    if importlib.util.find_spec("triton") is None:
        print("Triton not found; skipping torch.compile (eager mode)")
        return model

    if sys.platform == "win32":
        import torch._inductor.config as inductor_config

        # Backward kernel names (~200 chars) overflow MAX_PATH in the Triton cache.
        inductor_config.triton.descriptive_names = False
        # torch 2.6 os.replace()s onto an existing dir on FX-cache hits; Windows refuses.
        inductor_config.bundle_triton_into_fx_graph_cache = False

    return torch.compile(model, mode=mode, **kwargs)
