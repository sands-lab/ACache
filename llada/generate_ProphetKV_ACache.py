"""ProphetKV selector sidecar; the ACache generator is unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_ACache as acache
from prophetkv_selector import generate


def generate_with_prophetkv_anchor_attention(model, prompt, **kwargs):
    return generate(acache, "llada", model, prompt, **kwargs)
