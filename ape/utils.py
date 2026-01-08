import base64
from io import BytesIO
from typing import TYPE_CHECKING
from loguru import logger
from functools import lru_cache
from ape.logging import setup_logger  # re-export central function

if TYPE_CHECKING:  # pragma: no cover – typing only
    from PIL import Image  # noqa: F401

def decode_base64_image(image_base64: str) -> "Image.Image":
    try:
        from PIL import Image  # local import – heavy
    except ImportError as exc:
        raise RuntimeError(
            "The 'Pillow' library is required for image decoding. Install it via 'pip install pillow'."
        ) from exc

    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data)).convert("RGB")

def encode_image_base64(image: "Image.Image") -> str:
    try:
        from PIL import Image  # ensure Pillow present for type hints
    except ImportError as exc:
        raise RuntimeError(
            "The 'Pillow' library is required for image encoding. Install it via 'pip install pillow'."
        ) from exc

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str = "Qwen/Qwen3-8B"):
    """Return a cached *transformers* tokenizer instance.

    The cache avoids re-loading the same tokenizer weights repeatedly, which can
    be slow and memory-intensive.  We keep the cache small (8) because the
    framework typically only relies on one or two models simultaneously.
    """
    try:
        from transformers import AutoTokenizer  # local import to avoid global dependency at startup
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' library is required for token counting. Install it via 'pip install transformers'."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def count_tokens(text: str, model_name: str = "Qwen/Qwen3-8B") -> int:
    """Return the number of tokens *text* occupies for *model_name*.

    Examples
    --------
    >>> from ape.utils import count_tokens
    >>> count_tokens("Hello world")
    2

    If *transformers* is not installed an informative RuntimeError is raised.
    """
    tokenizer = _get_tokenizer(model_name)
    # NOTE: we do *not* add special tokens so the count reflects raw payload
    return len(tokenizer.encode(text, add_special_tokens=False))

async def get_ollama_model_info(model_name: str | None = None) -> dict:
    """Return structured information about an Ollama model.

    The helper calls ``ollama.show`` (sync) and extracts the most
    relevant fields so other subsystems (e.g. memory budgeting,
    advanced prompting) can consume them easily.

    Parameters
    ----------
    model_name:
        Model identifier understood by Ollama – if *None* the default
        from :data:`ape.settings.settings.LLM_MODEL` is used.

    Returns
    -------
    dict
        Canonical structure:: 

            {
              "model": "qwen3:8b",
              "architecture": "qwen3",
              "parameter_size": "8.2B",
              "context_length": 40960,
              "embedding_length": 4096,
              "quantization": "Q4_K_M",
              "capabilities": ["completion", "tools", "thinking"],
              "defaults": {"temperature": 0.6, "top_k": 20, ...},
            }

        Fields missing in the Ollama response are omitted.
    """

    from ape.settings import settings

    if model_name is None:
        model_name = settings.LLM_MODEL

    # Import lazily to avoid heavy deps where not needed
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The 'ollama' Python package is required to fetch model info. Install it with 'pip install ollama'."
        ) from exc

    try:
        client = ollama.AsyncClient(host=str(settings.OLLAMA_BASE_URL))
        raw = await client.show(model_name)
    except Exception as exc:
        # Enhance logging for "model not found" scenarios
        try:
            list_resp = await client.list()
            available = [m.get('name') for m in list_resp.get('models', [])]
            logger.error(f"Failed to find model '{model_name}'. Available models: {available}")
        except Exception:
            logger.error("Failed to list available models during error handling.")
        
        raise RuntimeError(f"Failed to fetch model info for '{model_name}': {exc}") from exc

    info: dict = {"model": model_name}

    # Attempt to normalise fields.  The exact keys vary slightly between
    # Ollama versions, so we probe multiple fallbacks.
    details = raw.get("details", {}) if isinstance(raw, dict) else {}

    # Helper to fetch a key ignoring spaces / underscores and case
    def _get(d: dict, *names):
        for n in names:
            if n in d:
                return d[n]
            # try flexible matching
            for k, v in d.items():
                if k.replace(" ", "_").lower() == n.lower():
                    return v
        return None

    info["architecture"] = _get(details, "architecture", "family")
    info["parameter_size"] = _get(details, "parameter_size", "parameters")
    info["context_length"] = _get(details, "context_length", "context_length_tokens")
    info["embedding_length"] = _get(details, "embedding_length", "embedding_size")
    info["quantization"] = _get(details, "quantization", "quantization_level")

    # ------------------------------------------------------------------
    # Fallback: parse raw.modelinfo for numeric limits if still missing
    # ------------------------------------------------------------------
    try:
        modelinfo_dict = getattr(raw, "modelinfo", {}) or {}
        if info.get("context_length") is None:
            for k, v in modelinfo_dict.items():
                if str(k).endswith("context_length") and isinstance(v, int):
                    info["context_length"] = v
                    break
        if info.get("embedding_length") is None:
            for k, v in modelinfo_dict.items():
                if str(k).endswith("embedding_length") and isinstance(v, int):
                    info["embedding_length"] = v
                    break
    except Exception:
        pass

    # capabilities may live at root or inside details – favour root
    caps = raw.get("capabilities") or details.get("capabilities")
    if caps:
        # ensure list[str]
        if isinstance(caps, (list, tuple)):
            info["capabilities"] = list(caps)
        elif isinstance(caps, str):
            info["capabilities"] = [c.strip() for c in caps.split()]

    # Default generation parameters (temperature, top_k, etc.)
    params = raw.get("parameters") or raw.get("defaults")
    if params and isinstance(params, dict):
        info["defaults"] = params

    # # licence text or pointer
    # lic = raw.get("license") or details.get("license")
    # if lic:
    #     # may be list of lines or long string
    #     if isinstance(lic, list):
    #         lic = "\n".join(lic)
    #     info["license"] = lic

    return info 