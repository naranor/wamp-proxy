# 🛠️ AI Agents Instructions (AGENTS.md)

This file describes the evolution and structure of the **Weighted Attention Message Pruner (WAMP)** project. It serves as a technical reference for both available context engines.

---

## 📌 Project Overview

WAMP is a research-oriented middleware for context compression. It implements a **Tri-modal Adaptive Engine** that uses semantic routing to select the optimal pruning policy for each user query.

### Core Architecture (V4.1)
The system supports multiple transformer-based encoders in ONNX format. Pruning is based on raw attention weights extracted from the model's final layers.

---

## 🧠 Engine Comparison & Technical Details

### 1. ModernBERT-base (Standard / Production)
*Best for long-context tasks (up to 8192 tokens).*
- **Architecture:** 22 layers, Rotary Embeddings, Unpadding.
- **WAMP Export:** Last 2 attention layers (20-21) to optimize memory.
- **Routing:** SetFit-trained classifier with 100% accuracy.
- **Capabilities:** High-precision analysis of complex agent trajectories.

**Validated Settings:**
| Category | Algorithm | Multiplier | Token Savings | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Needle** | `cls_max` | **0.98** | **38.3%** | 100% |
| **Reasoning** | `max_max` | **0.74** | **39.7%** | 100% |
| **Summary** | `max_max` | **1.00** | **55.7%** | 88% |

---

### 2. SetFit MiniLM-L12 (Compact / Fast)
*Best for low-resource environments and high speed.*
- **Architecture:** 12 layers, 512 token limit.
- **WAMP Export:** Full 12-layer attention matrix support.
- **Routing:** Original SetFit implementation (100% accuracy).
- **Capabilities:** Fast fact retrieval and conversational recaps.

**Validated Settings:**
| Category | Algorithm | Multiplier | Token Savings | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Needle** | `cls_max` | **0.99** | **28.6%** | 100% |
| **Reasoning** | `cls_max` | **0.95** | **0%** | 100% |
| **Summary** | `cls_max` | **0.99** | **36.8%** | 75% |

---

## 🗂 Project Structure

```text
wamp-proxy/
├── src/
│   └── wamp/
│       ├── api/            # FastAPI endpoints and proxy logic
│       ├── core/           # Logic: WAMPruner engine (SetFit), Config
│       └── __init__.py
├── benchmarks/             # Final validation suite (Needle, Multi-Doc, Coherence)
├── tools/                  # SetFit Trainer, ONNX Exporter, Quantizer, HF Uploader
├── tests/                  # Unit and API tests
├── main.py                 # 🚀 Entry point
├── requirements.txt        # Runtime dependencies
├── Makefile                # Developer shortcuts
└── README.md               # User and Research documentation
```

---

## 🧪 Research Evolution Context

1.  **DeBERTa-v3 (Legacy):** Initial research into tri-modal classification. Proved the "Cliff Effect" in attention pruning.
2.  **MiniLM-L12 (Unified V4):** Introduced 100% accurate SetFit routing.
3.  **ModernBERT (V4.1):** Solved context window limits and memory allocation issues via Sliding Window and optimized export.

---

*Last Updated: May 8, 2026 (Final Comparative Research Version)*
