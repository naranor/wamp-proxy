import logging
import json
import numpy as np
import onnxruntime as ort
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
from .config import (
    FILTER_MODEL_NAME,
    FILTER_MODEL_DIR,
    FILTER_MAX_TOKENS,
    FILTER_NEEDLE_MULT,
    FILTER_REASONING_MULT,
    FILTER_SUMMARY_MULT,
    FILTER_NEEDLE_ALGO,
    FILTER_REASONING_ALGO,
    FILTER_SUMMARY_ALGO,
)

logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class WAMPruner:
    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(FILTER_MODEL_DIR)
        self._ensure_model_exists()

        logger.info(f"Loading Unified SetFit Engine from {self.model_dir}...")

        # Load Tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()

        # Load ONNX model (Always use quantized if available)
        model_file = self.model_dir / "model_quantized.onnx"
        if not model_file.exists():
            model_file = self.model_dir / "model.onnx"

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_file), session_options)

        # Map outputs
        all_outputs = [o.name for o in self.session.get_outputs()]
        self.attention_layers = [o for o in all_outputs if o.startswith("attentions.")]
        self.attention_layers.sort(key=lambda x: int(x.split(".")[-1]))

        # Use up to 2 last layers for pruning
        self.pruning_layers = self.attention_layers[-2:]
        logger.info(f"Using attention layers for pruning: {self.pruning_layers}")

        # Load SetFit Classifier Head
        self.router_weights = None
        weights_path = self.model_dir / "router_weights_setfit.json"
        if weights_path.exists():
            with open(weights_path, "r") as f:
                self.router_weights = json.load(f)
            logger.info("SetFit Classification Head loaded successfully.")
        else:
            logger.warning(
                f"Classification weights not found in {self.model_dir}. Router disabled."
            )

        self._score_cache = {}

    def _ensure_model_exists(self):
        """Check if model files exist. Download from HF if missing."""
        required_files = ["tokenizer.json", "router_weights_setfit.json"]
        missing = [f for f in required_files if not (self.model_dir / f).exists()]

        if missing or (
            not (self.model_dir / "model_quantized.onnx").exists()
            and not (self.model_dir / "model.onnx").exists()
        ):
            logger.info(
                f"Model missing in {self.model_dir}. Attempting download: {FILTER_MODEL_NAME}..."
            )
            try:
                snapshot_download(
                    repo_id=FILTER_MODEL_NAME,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=[
                        "*.msgpack",
                        "*.h5",
                        "*.bin",
                        "*.pt",
                        "*.tflite",
                        "*.ot",
                        "model.onnx.data",
                    ],
                )
                logger.info(f"✓ Model {FILTER_MODEL_NAME} downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise RuntimeError(f"Model download failed: {e}")

    def classify_task(self, task: str) -> Tuple[str, float]:
        """Classify user task using SetFit (Mean Pooling + LogReg)."""
        if not self.router_weights:
            return "Summary", 1.0

        encoding = self.tokenizer.encode(task)
        # Use entire allowed window for routing
        ids = np.array([encoding.ids[:FILTER_MAX_TOKENS]], dtype=np.int64)
        mask = np.array([[1] * len(ids[0])], dtype=np.int64)

        outputs = self.session.run(None, {"input_ids": ids, "attention_mask": mask})

        last_hidden = outputs[0]
        embeddings = np.mean(last_hidden, axis=1)

        # LogReg Head
        coef = np.array(self.router_weights["coef"])
        intercept = np.array(self.router_weights["intercept"])
        scores = np.dot(embeddings, coef.T) + intercept

        # Softmax for probabilities
        exp_scores = np.exp(scores - np.max(scores))
        probs = (exp_scores / exp_scores.sum())[0]

        winner_idx = int(np.argmax(probs))
        inv_map = {0: "Summary", 1: "Needle", 2: "Reasoning"}

        label = inv_map[winner_idx]
        confidence = float(probs[winner_idx])

        logger.info(f"Task: '{task[:50]}...' -> Detected: {label} (p={confidence:.2f})")
        return label, confidence

    def get_attention_filtered(
        self, messages: List[Dict[str, Any]], task: str
    ) -> List[Dict[str, Any]]:
        """Filter messages using attention weights with sliding window logic."""
        if not messages:
            return []

        # 1. Route task
        task_type, _ = self.classify_task(task)

        # 2. Get Config (Multiplier & Algo)
        mult_map = {
            "Needle": float(FILTER_NEEDLE_MULT),
            "Reasoning": float(FILTER_REASONING_MULT),
            "Summary": float(FILTER_SUMMARY_MULT),
        }
        algo_map = {
            "Needle": FILTER_NEEDLE_ALGO,
            "Reasoning": FILTER_REASONING_ALGO,
            "Summary": FILTER_SUMMARY_ALGO,
        }

        multiplier = mult_map.get(task_type, 1.0)
        algo = algo_map.get(task_type, "cls_max")

        # 3. Calculate scores
        scores_dict = {}
        task_prompt = f"Analyze relevance to: '{task}'"
        
        task_enc = self.tokenizer.encode(task_prompt)
        task_ids = task_enc.ids
        
        # Ensure task doesn't leave less than a small buffer for message scan
        limit_for_task = max(1, FILTER_MAX_TOKENS - 128)
        if len(task_ids) > limit_for_task:
            task_ids = task_ids[:limit_for_task]
        task_len = len(task_ids)

        for i, msg in enumerate(messages):
            if i == 0 or i >= len(messages) - 2:
                continue

            content = self.get_content(msg)
            msg_key = hashlib.md5((task + content + algo).encode("utf-16")).hexdigest()

            if msg_key in self._score_cache:
                scores_dict[i] = self._score_cache[msg_key]
                continue

            msg_enc = self.tokenizer.encode(content)
            msg_ids = msg_enc.ids
            msg_len = len(msg_ids)

            # --- DYNAMIC SLIDING WINDOW ---
            window_size = FILTER_MAX_TOKENS - task_len
            if window_size <= 0:
                window_size = 1

            chunk_scores = []
            
            for start_idx in range(0, max(1, msg_len), window_size):
                end_idx = min(start_idx + window_size, msg_len)
                current_msg_chunk = msg_ids[start_idx:end_idx]
                
                if not current_msg_chunk:
                    break
                    
                input_ids = task_ids + current_msg_chunk
                ids_np = np.array([input_ids], dtype=np.int64)
                mask_np = np.ones_like(ids_np)

                outputs = self.session.run(None, {"input_ids": ids_np, "attention_mask": mask_np})

                importance = 0
                chunk_msg_len = len(current_msg_chunk)
                for layer_name in self.pruning_layers:
                    layer_idx = [o.name for o in self.session.get_outputs()].index(layer_name)
                    slc = outputs[layer_idx][0, :, :task_len, task_len : task_len + chunk_msg_len]
                    
                    if slc.size == 0:
                        continue

                    if algo == "mean_max":
                        importance += float(np.mean(np.max(slc, axis=2)))
                    elif algo == "cls_max":
                        importance += float(np.max(slc[:, 0, :]))
                    elif algo == "mean_mean":
                        importance += float(np.mean(slc))
                    elif algo == "max_max":
                        importance += float(np.max(slc))
                    else:
                        importance += float(np.max(slc[:, 0, :]))

                chunk_scores.append(importance / len(self.pruning_layers))
                
                if end_idx >= msg_len:
                    break

            score = max(chunk_scores) if chunk_scores else 0
            scores_dict[i] = score
            self._score_cache[msg_key] = score

        # 4. Prune
        all_scores = list(scores_dict.values())
        baseline = np.median(all_scores) if all_scores else 0
        threshold = baseline * multiplier

        keep_indices = {0}
        for i in range(max(0, len(messages) - 2), len(messages)):
            keep_indices.add(i)

        for i, msg in enumerate(messages):
            if i in keep_indices:
                continue
            if scores_dict.get(i, 0) >= threshold:
                keep_indices.add(i)

        filtered = [messages[i] for i in sorted(list(keep_indices))]

        logger.info(
            f"Filter ({task_type}, mult={multiplier}): {len(messages)} -> {len(filtered)} msgs (thr={threshold:.4f})"
        )
        return filtered

    def get_content(self, msg: Dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(
                [part.get("text", "") for part in content if part.get("type") == "text"]
            )
        return str(content)
