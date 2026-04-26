# Research Step 3: Reasoning Audit (114 msgs)
**Model:** DeBERTa-v3-small-NLI (INT8)

### Algorithm: mean_max
| Multiplier | Compression % | Recall (Logic) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.0% | 100% | ✅ PASS |
| 0.98 | 15.8% | 86% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 29% | ⚠️ PARTIAL |
| 1.02 | 81.6% | 14% | ⚠️ PARTIAL |
| 1.04 | 92.1% | 14% | ⚠️ PARTIAL |
| 1.06 | 94.7% | 14% | ⚠️ PARTIAL |
| 1.08 | 97.4% | 14% | ⚠️ PARTIAL |
| 1.10 | 97.4% | 14% | ⚠️ PARTIAL |

### Algorithm: max_max
| Multiplier | Compression % | Recall (Logic) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.0% | 100% | ✅ PASS |
| 0.98 | 1.8% | 100% | ✅ PASS |
| 1.00 | 48.2% | 86% | ⚠️ PARTIAL |
| 1.02 | 90.4% | 29% | ⚠️ PARTIAL |
| 1.04 | 97.4% | 14% | ⚠️ PARTIAL |
| 1.06 | 97.4% | 14% | ⚠️ PARTIAL |
| 1.08 | 97.4% | 14% | ⚠️ PARTIAL |
| 1.10 | 97.4% | 14% | ⚠️ PARTIAL |

### Algorithm: mean_mean
| Multiplier | Compression % | Recall (Logic) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.9% | 86% | ⚠️ PARTIAL |
| 0.92 | 1.8% | 71% | ⚠️ PARTIAL |
| 0.94 | 3.5% | 43% | ⚠️ PARTIAL |
| 0.96 | 14.9% | 43% | ⚠️ PARTIAL |
| 0.98 | 37.7% | 29% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 29% | ⚠️ PARTIAL |
| 1.02 | 64.0% | 29% | ⚠️ PARTIAL |
| 1.04 | 80.7% | 29% | ⚠️ PARTIAL |
| 1.06 | 88.6% | 29% | ⚠️ PARTIAL |
| 1.08 | 89.5% | 29% | ⚠️ PARTIAL |
| 1.10 | 89.5% | 29% | ⚠️ PARTIAL |

### Algorithm: cls_max
| Multiplier | Compression % | Recall (Logic) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 10.5% | 100% | ✅ PASS |
| 0.92 | 22.8% | 100% | ✅ PASS |
| 0.94 | 36.8% | 86% | ⚠️ PARTIAL |
| 0.96 | 42.1% | 57% | ⚠️ PARTIAL |
| 0.98 | 46.5% | 43% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 43% | ⚠️ PARTIAL |
| 1.02 | 61.4% | 29% | ⚠️ PARTIAL |
| 1.04 | 70.2% | 29% | ⚠️ PARTIAL |
| 1.06 | 80.7% | 29% | ⚠️ PARTIAL |
| 1.08 | 87.7% | 14% | ⚠️ PARTIAL |
| 1.10 | 88.6% | 14% | ⚠️ PARTIAL |

