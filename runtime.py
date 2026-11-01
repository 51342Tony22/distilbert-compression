# 用于管理项目路径设备选择模型缓存和量化后端
import platform
from pathlib import Path

import torch


FINAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = FINAL_DIR / "outputs"
SOURCE_DATA_OUTPUT_DIR = OUTPUT_DIR / "source_data"
FINAL_EXPERIMENT_NAME = (
    "final_4layers_l1headpruned_cosine_aug_wd0001_bs32"
)
BASE_MODEL_NAME = "distilbert-base-uncased"


def from_pretrained_local_first(factory, **kwargs):
    """优先使用本地 Hugging Face 缓存，缺失时再尝试联网"""

    try:
        return factory.from_pretrained(
            BASE_MODEL_NAME,
            local_files_only=True,
            **kwargs,
        )
    except OSError:
        return factory.from_pretrained(
            BASE_MODEL_NAME,
            **kwargs,
        )


def load_distilbert_tokenizer():
    from transformers import DistilBertTokenizer

    return from_pretrained_local_first(DistilBertTokenizer)


def get_training_device():
    """按 CUDA、Apple MPS、CPU 的顺序选择训练设备,因为后续使用mac，所以会优先MPS"""

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)

    if (
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def configure_quantized_backend():
    """为当前机器选择可用的动态量化后端"""

    supported_engines = set(torch.backends.quantized.supported_engines)
    machine = platform.machine().lower()

    if machine in {"arm64", "aarch64"}:
        preferred_engines = ["qnnpack", "x86", "fbgemm"]
    else:
        preferred_engines = ["x86", "fbgemm", "qnnpack"]

    for engine in preferred_engines:
        if engine in supported_engines:
            torch.backends.quantized.engine = engine
            return engine

    raise RuntimeError(
        "No supported PyTorch quantization backend was found. "
        f"Available engines: {sorted(supported_engines)}"
    )
