import logging
import json
import numpy as np
import onnxruntime as ort
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tokenizers import Tokenizer
from .config import (
    FILTER_MODEL_NAME,
    FILTER_MODEL_DIR,
    FILTER_MAX_TOKENS,
    FILTER_NEEDLE_MULT,
    FILTER_REASONING_MULT,
    FILTER_SUMMARY_MULT,
)

logger = logging.getLogger(__name__)


class WAMPruner:
    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else Path(FILTER_MODEL_DIR)
        self._ensure_model_exists()

        logger.info(f"Loading {FILTER_MODEL_NAME} from {self.model_dir}...")

        # Load official tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=FILTER_MAX_TOKENS)

        # Load ONNX model
        model_file = self.model_dir / "model_quantized.onnx"
        if not model_file.exists():
            model_file = self.model_dir / "model.onnx"

        session_options = ort.SessionOptions()
        import os

        cpus = os.cpu_count() or 4
        session_options.intra_op_num_threads = cpus
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_file), session_options)

        # Automatically detect available attention outputs
        all_outputs = [o.name for o in self.session.get_outputs()]
        self.output_names = [o for o in all_outputs if o.startswith("attentions.")]
        self.output_names.sort(key=lambda x: int(x.split(".")[-1]))

        # Pick the last 2 layers
        if len(self.output_names) > 2:
            self.output_names = self.output_names[-2:]

        self._score_cache = {}

        # Load NLI/Needle weights
        self.router_weights = None
        self.needle_weights = None

        core_path = Path(__file__).parent
        if (core_path / "router_weights.json").exists():
            with open(core_path / "router_weights.json", "r") as f:
                self.router_weights = json.load(f)
        if (core_path / "needle_weights.json").exists():
            with open(core_path / "needle_weights.json", "r") as f:
                self.needle_weights = json.load(f)

        logger.info("Adaptive router and Needle detector loaded successfully.")

    def _get_cache_key(self, task: str, content: str, algo: str) -> str:
        hasher = hashlib.md5()
        hasher.update(task.encode("utf-16"))
        hasher.update(content.encode("utf-16"))
        hasher.update(algo.encode("utf-16"))
        return hasher.hexdigest()

    def _ensure_model_exists(self):
        """Check if model files exist. Download from HF if missing."""
        required_files = ["tokenizer.json", "model_quantized.onnx"]
        missing = [f for f in required_files if not (self.model_dir / f).exists()]
        
        # Fallback check for unquantized model
        if not (self.model_dir / "model_quantized.onnx").exists() and (self.model_dir / "model.onnx").exists():
            if not (self.model_dir / "tokenizer.json").exists():
                missing = ["tokenizer.json"]
            else:
                return

        if missing:
            logger.info(f"Model files {missing} missing in {self.model_dir}. Attempting download from HF: {FILTER_MODEL_NAME}...")
            try:
                snapshot_download(
                    repo_id=FILTER_MODEL_NAME,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.pt", "*.tflite", "*.ot"]
                )
                logger.info(f"✓ Model {FILTER_MODEL_NAME} downloaded successfully to {self.model_dir}")
            except Exception as e:
                error_msg = f"Failed to download model {FILTER_MODEL_NAME}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    def get_content(self, msg: Dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, list):
            return " ".join(
                [part.get("text", "") for part in content if part.get("type") == "text"]
            )
        return str(content)

    def classify_task(self, task_text: str) -> str:
        """Hierarchy: Stage 1: Reasoning, Stage 2: Needle, Stage 3: Summary"""
        from scipy.stats import kurtosis

        encoding = self.tokenizer.encode(task_text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        outputs = self.session.run(None, {"input_ids": ids, "attention_mask": mask})

        # Extract Composite Features (CLS + Mean + Kurtosis + Max + Length)
        last_hidden = outputs[0][0]
        cls_emb = last_hidden[0]
        mean_emb = np.mean(last_hidden, axis=0)
        last_attn = outputs[-1][0]
        avg_attn = np.mean(last_attn, axis=0)
        flat_attn = avg_attn.flatten()
        attn_kurt = kurtosis(flat_attn)
        attn_max = np.max(flat_attn)
        features = np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, len(encoding.ids)]])

        # --- Stage 1: Reasoning Detection (High Priority) ---
        if self.router_weights:
            summary_keywords = [
                "summarize",
                "summary",
                "tldr",
                "overview",
                "recap",
                "synopsis",
                "condense",
                "briefly",
            ]
            if any(kw in task_text.lower() for kw in summary_keywords):
                return "Summary"

            w_r, b_r = (
                np.array(self.router_weights["coef"]),
                np.array(self.router_weights["intercept"]),
            )
            prob_reasoning = 1 / (1 + np.exp(-(np.dot(w_r, features) + b_r)[0]))
            if prob_reasoning > 0.5:
                logger.info(f"Stage 1: Reasoning detected (p={prob_reasoning:.2f})")
                return "Reasoning"

        # --- Stage 2: Needle Detection ---
        if self.needle_weights:
            w_n, b_n = (
                np.array(self.needle_weights["coef"]),
                np.array(self.needle_weights["intercept"]),
            )
            prob_needle = 1 / (1 + np.exp(-(np.dot(w_n, features) + b_n)[0]))

            # Set threshold to 0.7 for balance
            if prob_needle > 0.5:
                logger.info(f"Stage 2: Needle confirmed (p={prob_needle:.2f})")
                return "Needle"

        return "Summary"

    def get_attention_filtered(
        self,
        messages: List[Dict[str, Any]],
        task: str,
        original_query: str = None,
        keep_last_n: int = 4,
    ) -> Tuple[List[Dict[str, Any]], List[float], float, float]:
        import time

        t_start = time.time()

        # Classify based on the raw user query if provided, otherwise the full task
        query_to_classify = original_query if original_query else task
        task_category = self.classify_task(query_to_classify)

        # Scientific Optimized Policy (Configurable via .env)
        if task_category == "Reasoning":
            mult, algo = FILTER_REASONING_MULT, "cls_max"
        elif task_category == "Needle":
            mult, algo = FILTER_NEEDLE_MULT, "mean_max"
        else:
            mult, algo = FILTER_SUMMARY_MULT, "mean_mean"

        logger.info(f"Policy: {task_category} | {algo} | mult={mult}")

        total = len(messages)
        if total <= 2:
            return messages, [], 0.0, 0.0

        always_keep = {0}
        for i in range(max(1, total - keep_last_n), total):
            always_keep.add(i)

        task_enc = self.tokenizer.encode(f"TASK: {task}\n\n")
        task_ids, task_len = task_enc.ids, len(task_enc.ids)

        all_scores_dict = {}
        batch_queue = []

        for idx in range(total):
            if idx in always_keep:
                all_scores_dict[idx] = 1.0
                continue

            content = self.get_content(messages[idx])
            cache_key = self._get_cache_key(task, content, algo)

            if cache_key in self._score_cache:
                all_scores_dict[idx] = self._score_cache[cache_key]
            else:
                msg_ids = self.tokenizer.encode(f"Msg: {content}\n").ids[: 128 - task_len]
                batch_queue.append(
                    {
                        "idx": idx,
                        "ids": msg_ids,
                        "len": len(msg_ids),
                        "key": cache_key,
                        "algo": algo,
                    }
                )

            if len(batch_queue) >= 32:
                self._process_batch(batch_queue, task_ids, task_len, all_scores_dict)
                batch_queue = []

        if batch_queue:
            self._process_batch(batch_queue, task_ids, task_len, all_scores_dict)

        all_scores = [all_scores_dict.get(i, 0.0) for i in range(total)]
        filterable = [all_scores[i] for i in range(total) if i not in always_keep]
        baseline = float(np.median(filterable)) if filterable else 0.0
        threshold = baseline * mult

        filtered = [
            messages[i] for i in range(total) if i in always_keep or all_scores[i] >= threshold
        ]
        logger.info(f"Done in {time.time() - t_start:.3f}s")
        return filtered, all_scores, threshold, baseline

    def _process_batch(self, batch_data, task_ids, task_len, scores_dict):
        batch_size = len(batch_data)
        max_m_len = max(item["len"] for item in batch_data)
        input_ids = np.zeros((batch_size, task_len + max_m_len), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)

        for i, item in enumerate(batch_data):
            seq = task_ids + item["ids"]
            input_ids[i, : len(seq)] = seq
            attention_mask[i, : len(seq)] = 1

        outputs = self.session.run(
            self.output_names, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        for i, item in enumerate(batch_data):
            importance = 0.0
            for layer_attn in outputs:
                # slice: (heads, task_tokens, msg_tokens)
                slc = layer_attn[i, :, :task_len, task_len : task_len + item["len"]]
                if slc.size == 0:
                    continue

                algo = item["algo"]
                if algo == "mean_max":
                    importance += float(np.mean(np.max(slc, axis=-1)))
                elif algo == "max_max":
                    importance += float(np.max(slc))
                elif algo == "mean_mean":
                    importance += float(np.mean(slc))
                elif algo == "cls_max":
                    importance += float(np.max(slc[:, 0, :]))  # Attention FROM [CLS] to msg

            score = importance / len(self.output_names)
            scores_dict[item["idx"]] = score
            self._score_cache[item["key"]] = score
