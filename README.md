# Weighted Attention Message Pruner (WAMP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

> **⚠️ RESEARCH PROTOTYPE / PoC**  
> **WAMP** is an experimental tool exploring the use of attention weights for context pruning. It features a **Tri-modal Adaptive Engine** that uses semantic routing to select optimal pruning strategies.

## 📌 Overview

**WAMP** is an intelligent middleware for research into LLM context optimization. It supports multiple encoder backbones (ModernBERT, MiniLM) and implements dynamic attention-based policies to prune redundant messages while preserving semantic integrity.

## 📦 Hugging Face Resources

- **Primary Engine:** [ModernBERT Multilingual ONNX Router](https://huggingface.co/naranor/SetFit-ModernBERT-WAMP-V1) (8192 context window)
- **Compact Engine:** [SetFit Multilingual ONNX Router V1](https://huggingface.co/naranor/SetFit-Multilingual-ONNX-Router-V1) (Fast, low memory)
- **Training Data:** [WAMP Router Intent Dataset](https://huggingface.co/datasets/naranor/WAMP-Router-Intent-Dataset)

## 📊 Research Benchmarks: Engine Comparison

WAMP supports two high-performance engines. Selection depends on hardware constraints and context length requirements.

### 1. ModernBERT-base (Standard / Long-Context)
*Best for complex reasoning and large messages.*

| Scenario | Algorithm | Multiplier | Token Savings | Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fact Retrieval** | `cls_max` | 0.98 | **~38%** | **100%** | ✅ SAFE |
| **Reasoning** | `max_max` | 0.74 | **~40%** | **100%** | ✅ SAFE |
| **Summary** | `max_max` | 1.00 | **~56%** | **88%** | ✅ POWER |

### 2. SetFit MiniLM-L12 (Lightweight / Fast)
*Best for high-speed simple tasks.*

| Scenario | Algorithm | Multiplier | Token Savings | Recall | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fact Retrieval** | `cls_max` | 0.99 | **~29%** | **100%** | ✅ SAFE |
| **Reasoning** | `cls_max` | 0.95 | **0%** | **100%** | ✅ SAFE |
| **Summary** | `cls_max` | 0.99 | **~37%** | **75%** | ⚠️ FLOW |

## 🧠 Switching Engines

Update your `.env` file to select the desired model:

```env
# ModernBERT (Standard)
FILTER_MODEL_DIR=./model_modernbert_onnx
FILTER_MAX_TOKENS=2048

# MiniLM (Fast)
# FILTER_MODEL_DIR=./model_setfit_onnx
# FILTER_MAX_TOKENS=512
```

## 🏗️ Architecture

```mermaid
graph TD
    User([User App]) -->|Full Context| WAMP[WAMP Proxy]
    WAMP -->|1. Intent Routing| Router{Semantic Router}
    Router -->|Needle| P1[Precise Strategy]
    Router -->|Logic| P2[Logical Strategy]
    Router -->|Summary| P3[Broad Strategy]
    P1 & P2 & P3 -->|2. Pruning| Final[Compressed Context]
    Final -->|3. Forwarding| Upstream[[LLM Upstream]]
    Upstream -->|Response| User
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/naranor/wamp-proxy.git
cd wamp-proxy
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
python main.py
```

## 🛠 Research Tools
- `tools/research_modernbert_filter.py` — Granular research into ModernBERT multipliers.
- `tools/research_setfit_filter.py` — Research into MiniLM multipliers.
- `benchmarks/live_proxy_test.py` — End-to-end verification.

---
*Created for the research of Attention mechanisms in Transformer architectures.*
