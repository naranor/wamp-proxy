import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

# Server settings
PORT = int(os.getenv("PORT", "3000"))
HOST = os.getenv("HOST", "0.0.0.0")
UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://api.openai.com").rstrip("/")

# Filter settings
ENABLE_ATTENTION_FILTER = os.getenv("ENABLE_ATTENTION_FILTER", "true").lower() == "true"
FILTER_MODEL_NAME = os.getenv("HF_MODEL_NAME", "naranor/ModernBERT-base-ONNX-Attentions")
FILTER_MODEL_DIR = os.getenv("FILTER_MODEL_DIR", str(BASE_DIR / "modernbert_model"))
FILTER_THRESHOLD_MULTIPLIER = float(os.getenv("FILTER_THRESHOLD_MULTIPLIER", "1.5"))
FILTER_MAX_TOKENS = int(os.getenv("FILTER_MAX_TOKENS", "1024"))
FILTER_KEEP_LAST_N = int(os.getenv("FILTER_KEEP_LAST_N", "4"))

# Adaptive Category Multipliers
FILTER_NEEDLE_MULT = float(os.getenv("FILTER_NEEDLE_MULT", "0.98"))
FILTER_REASONING_MULT = float(os.getenv("FILTER_REASONING_MULT", "1.02"))
FILTER_SUMMARY_MULT = float(os.getenv("FILTER_SUMMARY_MULT", "1.05"))

# Pruning Algorithms (mean_max, cls_max, mean_mean, max_max)
FILTER_NEEDLE_ALGO = os.getenv("FILTER_NEEDLE_ALGO", "cls_max")
FILTER_REASONING_ALGO = os.getenv("FILTER_REASONING_ALGO", "cls_max")
FILTER_SUMMARY_ALGO = os.getenv("FILTER_SUMMARY_ALGO", "cls_max")
