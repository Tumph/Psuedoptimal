"""Model loading utilities with optional 4-bit quantization and LoRA."""

import os
import platform
from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Check if bitsandbytes is available (Linux + CUDA only)
BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig

    BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None

# Check platform
IS_MACOS = platform.system() == "Darwin"
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device_info() -> dict:
    """Get information about available compute devices."""
    return {
        "platform": platform.system(),
        "cuda_available": HAS_CUDA,
        "mps_available": HAS_MPS,
        "bnb_available": BNB_AVAILABLE,
        "recommended_device": "cuda" if HAS_CUDA else ("mps" if HAS_MPS else "cpu"),
    }


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_use_double_quant: bool = True,
):
    """
    Create BitsAndBytesConfig for 4-bit quantization if available.

    Returns None if bitsandbytes is not available.
    """
    if not BNB_AVAILABLE:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
    bias: str = "none",
) -> LoraConfig:
    """
    Create LoRA configuration for efficient fine-tuning.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to
        bias: Bias training mode

    Returns:
        LoraConfig instance
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )


def _get_device_map_and_dtype():
    """Determine optimal device_map and dtype for current platform."""
    if HAS_CUDA:
        return "auto", torch.bfloat16
    elif HAS_MPS:
        # MPS has numerical stability issues with float16 during generation
        # Use float32 for stable training (uses more memory but avoids NaN/Inf)
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


def _get_hf_token(hf_token: Optional[str] = None) -> Optional[str]:
    """Get HuggingFace token from parameter or environment."""
    if hf_token:
        return hf_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_student_model(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    quantization_config=None,
    lora_config: Optional[LoraConfig] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_quantization: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the Student model (policy) with optional 4-bit quantization and LoRA.

    On macOS or without bitsandbytes, falls back to full precision.

    Args:
        model_name: HuggingFace model identifier
        quantization_config: BitsAndBytes config (auto-detected if None)
        lora_config: LoRA config (uses default if None)
        device_map: Device placement strategy (auto-detected if None)
        torch_dtype: Model dtype (auto-detected if None)
        use_quantization: Whether to use quantization (if available)
        hf_token: HuggingFace token for gated models (or set HF_TOKEN env var)

    Returns:
        Tuple of (model, tokenizer)
    """
    token = _get_hf_token(hf_token)
    # Auto-detect device and dtype
    if device_map is None or torch_dtype is None:
        auto_device, auto_dtype = _get_device_map_and_dtype()
        device_map = device_map or auto_device
        torch_dtype = torch_dtype or auto_dtype

    # Determine if we can use quantization
    can_quantize = use_quantization and BNB_AVAILABLE and HAS_CUDA

    if lora_config is None:
        lora_config = get_lora_config()

    print(f"Loading Student model: {model_name}")
    if can_quantize:
        print(f"  Quantization: 4-bit NF4")
        if quantization_config is None:
            quantization_config = get_quantization_config()
    else:
        print(f"  Quantization: None (bitsandbytes {'not available' if not BNB_AVAILABLE else 'requires CUDA'})")
        print(f"  Precision: {torch_dtype}")
        quantization_config = None

    print(f"  Device: {device_map}")
    print(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=torch_dtype,
        trust_remote_code=True,
        token=token,
        attn_implementation="eager",  # For compatibility
    )

    # Prepare for k-bit training if quantized
    if can_quantize:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )

    # Set padding token and side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batch generation

    return model, tokenizer


def load_generator_model(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    quantization_config=None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_quantization: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the Generator model (frozen interpreter).

    On macOS or without bitsandbytes, falls back to full precision.

    Args:
        model_name: HuggingFace model identifier
        quantization_config: BitsAndBytes config (auto-detected if None)
        device_map: Device placement strategy (auto-detected if None)
        torch_dtype: Model dtype (auto-detected if None)
        use_quantization: Whether to use quantization (if available)
        hf_token: HuggingFace token for gated models (or set HF_TOKEN env var)

    Returns:
        Tuple of (model, tokenizer)
    """
    token = _get_hf_token(hf_token)

    # Auto-detect device and dtype
    if device_map is None or torch_dtype is None:
        auto_device, auto_dtype = _get_device_map_and_dtype()
        device_map = device_map or auto_device
        torch_dtype = torch_dtype or auto_dtype

    # Determine if we can use quantization
    can_quantize = use_quantization and BNB_AVAILABLE and HAS_CUDA

    print(f"Loading Generator model: {model_name}")
    if can_quantize:
        print(f"  Quantization: 4-bit NF4")
        if quantization_config is None:
            quantization_config = get_quantization_config()
    else:
        print(f"  Quantization: None (bitsandbytes {'not available' if not BNB_AVAILABLE else 'requires CUDA'})")
        print(f"  Precision: {torch_dtype}")
        quantization_config = None

    print(f"  Device: {device_map}")
    print(f"  Mode: Frozen (no gradients)")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=torch_dtype,
        trust_remote_code=True,
        token=token,
        attn_implementation="eager",  # For compatibility
    )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set to eval mode
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models

    return model, tokenizer


def estimate_memory_usage(
    student_size_b: float = 0.5,
    generator_size_b: float = 1.5,
    quantization: str = "auto",
    lora_rank: int = 16,
) -> dict:
    """
    Estimate memory usage for the dual-model setup.

    Args:
        student_size_b: Student model size in billions of params
        generator_size_b: Generator model size in billions of params
        quantization: Quantization level ("4bit", "8bit", "16bit", "auto")
        lora_rank: LoRA rank for Student model

    Returns:
        Dict with memory estimates in GB
    """
    # Auto-detect quantization based on availability
    if quantization == "auto":
        if BNB_AVAILABLE and HAS_CUDA:
            quantization = "4bit"
        elif HAS_MPS:
            quantization = "32bit"  # MPS uses float32 for stability
        else:
            quantization = "16bit"

    bytes_per_param = {"4bit": 0.5, "8bit": 1.0, "16bit": 2.0, "32bit": 4.0}
    bpp = bytes_per_param.get(quantization, 2.0)

    student_base = student_size_b * bpp
    generator_base = generator_size_b * bpp

    # LoRA parameters (trainable)
    lora_params = student_size_b * 0.01 * (lora_rank / 16)  # ~1% of params
    lora_memory = lora_params * 2  # 16-bit

    # Optimizer states (Adam: 2 states per param)
    optimizer_memory = lora_memory * 2

    # Gradients
    gradient_memory = lora_memory

    # Activations (rough estimate)
    activation_memory = 2.0

    total = (
        student_base
        + generator_base
        + lora_memory
        + optimizer_memory
        + gradient_memory
        + activation_memory
    )

    return {
        "quantization": quantization,
        "student_model_gb": student_base,
        "generator_model_gb": generator_base,
        "lora_adapter_gb": lora_memory,
        "optimizer_states_gb": optimizer_memory,
        "gradients_gb": gradient_memory,
        "activations_gb": activation_memory,
        "total_estimated_gb": total,
    }


def clear_memory():
    """Force clear memory cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif HAS_MPS:
        torch.mps.empty_cache()
