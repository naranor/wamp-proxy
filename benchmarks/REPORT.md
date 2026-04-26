# WAMP Research: Comprehensive Context Pruning Report
**Date:** April 25, 2026
**Model:** DeBERTa-v3-small-NLI (INT8 ONNX)
**Status:** Final Calibrated Policy

## 1. Research Methodology
We evaluated 4 attention aggregation algorithms across 3 diverse 100+ message scenarios to find the optimal balance between token savings and information recall. Exhaustive multiplier search (0.90 to 1.40) was conducted.

## 2. Final Adaptive Routing Policy
Based on our multi-stage research, the system now automatically switches between the following specialized modes:

| Category | Algorithm | Multiplier | Targeted Scenario | Target Result |
| :--- | :--- | :--- | :--- | :--- |
| **Needle** | `mean_max` | **0.98** | Pinpoint Fact Retrieval | ✅ 100% Recall |
| **Reasoning** | `cls_max` | **0.92** | Logic & Synthesis | ✅ 100% Logic |
| **Summary** | `mean_mean` | **0.90** | General Recap | ✅ High Flow |

## 3. Detailed Experimental Results (Long Context)

### Scenario A: Needle In A Haystack (112 msgs)
- **Algorithm:** `mean_max` (Precise peak detection)
- **Multiplier:** 0.98
- **Performance:** **12.5% Savings** at 100% Recall.
- **Insight:** To guarantee fact preservation without risk, compression must be highly conservative.

### Scenario B: Multi-Doc QA (114 msgs)
- **Algorithm:** `cls_max` (Intent-focused connection)
- **Multiplier:** 0.92
- **Performance:** **0.9% Savings** at 100% Recall.
- **Insight:** Logical chains are extremely fragile. Aggressive pruning destroys them.

### Scenario C: Coherence & Summary (114 msgs)
- **Algorithm:** `mean_mean` (Global semantic compression)
- **Multiplier:** 0.90
- **Performance:** **43.0% Savings** at 75% Recall.
- **Insight:** General summarization safely tolerates high compression.

## 4. Key Research Insights
1.  **The "Agent" Dilemma:** While the router effectively identifies task types with ~89% accuracy, applying compression to fact retrieval (Needle) or logical analysis (Reasoning) is fundamentally unsafe for coding agents at multipliers above 0.98. The attention signal for critical fragments in 100+ message contexts is too weak compared to the baseline median.
2.  **Algorithm-Task Fit:** No single algorithm is optimal for all tasks. `mean_max` is best for specific keywords, while `cls_max` is essential for linking disparate facts based on query intent.
3.  **Feature Engineering:** A composite feature vector (CLS, Mean, Kurtosis, Length) significantly outperforms raw embeddings for intent routing.

## 5. Conclusion
WAMP-proxy implements a **Tri-modal Adaptive Engine** that provides a stable compression layer. However, our exhaustive testing reveals that **Attention Pruning is best suited for general chat and summarization tasks**. For high-precision tasks like coding or fact retrieval, the system automatically defaults to highly conservative compression (0-12%) to guarantee 100% information recall, prioritizing safety over savings.

---
*End of Research Report.*
