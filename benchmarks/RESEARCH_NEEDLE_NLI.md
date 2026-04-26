# Research Step 2: Needle In A Haystack Audit (112 msgs)
**Model:** DeBERTa-v3-small-NLI (INT8)

### Algorithm: mean_max
| Multiplier | Compression % | Recall (Fact) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 1.8% | 100% | ✅ PASS |
| 0.98 | 12.5% | 100% | ✅ PASS |
| 1.00 | 47.3% | 100% | ✅ PASS |
| 1.02 | 83.0% | 50% | ⚠️ PARTIAL |
| 1.04 | 91.1% | 50% | ⚠️ PARTIAL |
| 1.06 | 96.4% | 0% | ❌ FAIL |
| 1.08 | 97.3% | 0% | ❌ FAIL |
| 1.10 | 97.3% | 0% | ❌ FAIL |

### Algorithm: max_max
| Multiplier | Compression % | Recall (Fact) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 0.0% | 100% | ✅ PASS |
| 0.92 | 0.0% | 100% | ✅ PASS |
| 0.94 | 0.0% | 100% | ✅ PASS |
| 0.96 | 0.0% | 100% | ✅ PASS |
| 0.98 | 0.0% | 100% | ✅ PASS |
| 1.00 | 48.2% | 0% | ❌ FAIL |
| 1.02 | 97.3% | 0% | ❌ FAIL |
| 1.04 | 97.3% | 0% | ❌ FAIL |
| 1.06 | 97.3% | 0% | ❌ FAIL |
| 1.08 | 97.3% | 0% | ❌ FAIL |
| 1.10 | 97.3% | 0% | ❌ FAIL |

### Algorithm: mean_mean
| Multiplier | Compression % | Recall (Fact) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 48.2% | 50% | ⚠️ PARTIAL |
| 0.92 | 48.2% | 50% | ⚠️ PARTIAL |
| 0.94 | 48.2% | 50% | ⚠️ PARTIAL |
| 0.96 | 48.2% | 50% | ⚠️ PARTIAL |
| 0.98 | 48.2% | 50% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 50% | ⚠️ PARTIAL |
| 1.02 | 50.9% | 50% | ⚠️ PARTIAL |
| 1.04 | 52.7% | 50% | ⚠️ PARTIAL |
| 1.06 | 54.5% | 50% | ⚠️ PARTIAL |
| 1.08 | 57.1% | 50% | ⚠️ PARTIAL |
| 1.10 | 59.8% | 50% | ⚠️ PARTIAL |

### Algorithm: cls_max
| Multiplier | Compression % | Recall (Fact) % | Status |
| :--- | :--- | :--- | :--- |
| 0.90 | 6.2% | 100% | ✅ PASS |
| 0.92 | 17.9% | 50% | ⚠️ PARTIAL |
| 0.94 | 25.0% | 50% | ⚠️ PARTIAL |
| 0.96 | 40.2% | 50% | ⚠️ PARTIAL |
| 0.98 | 43.8% | 50% | ⚠️ PARTIAL |
| 1.00 | 48.2% | 50% | ⚠️ PARTIAL |
| 1.02 | 51.8% | 50% | ⚠️ PARTIAL |
| 1.04 | 57.1% | 50% | ⚠️ PARTIAL |
| 1.06 | 70.5% | 50% | ⚠️ PARTIAL |
| 1.08 | 78.6% | 0% | ❌ FAIL |
| 1.10 | 86.6% | 0% | ❌ FAIL |

