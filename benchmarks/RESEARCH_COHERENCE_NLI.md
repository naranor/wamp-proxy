# Research Step 4: Coherence Audit (114 msgs)
**Model:** DeBERTa-v3-small-NLI (INT8)

### Algorithm: mean_max
| Multiplier | Compression % | Recall (Structure) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.0% | 100% | ✅ PASS |
| 0.98 | 14.9% | 62% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 62% | ⚠️ PARTIAL |
| 1.02 | 80.7% | 12% | ⚠️ PARTIAL |
| 1.04 | 92.1% | 0% | ❌ FAIL |
| 1.06 | 97.4% | 0% | ❌ FAIL |
| 1.08 | 97.4% | 0% | ❌ FAIL |
| 1.10 | 97.4% | 0% | ❌ FAIL |

### Algorithm: max_max
| Multiplier | Compression % | Recall (Structure) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.0% | 100% | ✅ PASS |
| 0.98 | 6.1% | 100% | ✅ PASS |
| 1.00 | 48.2% | 50% | ⚠️ PARTIAL |
| 1.02 | 92.1% | 0% | ❌ FAIL |
| 1.04 | 97.4% | 0% | ❌ FAIL |
| 1.06 | 97.4% | 0% | ❌ FAIL |
| 1.08 | 97.4% | 0% | ❌ FAIL |
| 1.10 | 97.4% | 0% | ❌ FAIL |

### Algorithm: mean_mean
| Multiplier | Compression % | Recall (Structure) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 43.0% | 75% | ⚠️ PARTIAL |
| 0.92 | 43.9% | 62% | ⚠️ PARTIAL |
| 0.94 | 43.9% | 62% | ⚠️ PARTIAL |
| 0.96 | 44.7% | 50% | ⚠️ PARTIAL |
| 0.98 | 44.7% | 50% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 50% | ⚠️ PARTIAL |
| 1.02 | 79.8% | 50% | ⚠️ PARTIAL |
| 1.04 | 87.7% | 38% | ⚠️ PARTIAL |
| 1.06 | 87.7% | 38% | ⚠️ PARTIAL |
| 1.08 | 87.7% | 38% | ⚠️ PARTIAL |
| 1.10 | 87.7% | 38% | ⚠️ PARTIAL |

### Algorithm: cls_max
| Multiplier | Compression % | Recall (Structure) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.9% | 100% | ✅ PASS |
| 0.98 | 8.8% | 100% | ✅ PASS |
| 1.00 | 48.2% | 62% | ⚠️ PARTIAL |
| 1.02 | 69.3% | 50% | ⚠️ PARTIAL |
| 1.04 | 86.8% | 50% | ⚠️ PARTIAL |
| 1.06 | 87.7% | 38% | ⚠️ PARTIAL |
| 1.08 | 89.5% | 38% | ⚠️ PARTIAL |
| 1.10 | 91.2% | 38% | ⚠️ PARTIAL |

