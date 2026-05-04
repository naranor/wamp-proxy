# Research Step 6: SetFit (MiniLM-L12) as Pruner

### Scenario: Needle (Fact)
| Algo | Multiplier | Savings % | Recall % | Status |
| :--- | :--- | :--- | :--- | :--- |
| mean_max | 0.90 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.91 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.92 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.93 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.94 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.95 | 9.1% | 100% | ✅ PASS |
| mean_max | 0.96 | 10.1% | 100% | ✅ PASS |
| mean_max | 0.97 | 26.2% | 50% | ❌ FAIL |
| mean_max | 0.98 | 35.3% | 50% | ❌ FAIL |
| mean_max | 0.99 | 52.7% | 50% | ❌ FAIL |
| mean_max | 1.00 | 60.6% | 50% | ❌ FAIL |
| cls_max | 0.90 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.91 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.92 | 1.1% | 100% | ✅ PASS |
| cls_max | 0.93 | 1.7% | 100% | ✅ PASS |
| cls_max | 0.94 | 2.3% | 100% | ✅ PASS |
| cls_max | 0.95 | 3.4% | 100% | ✅ PASS |
| cls_max | 0.96 | 11.8% | 100% | ✅ PASS |
| cls_max | 0.97 | 13.4% | 100% | ✅ PASS |
| cls_max | 0.98 | 24.0% | 100% | ✅ PASS |
| cls_max | 0.99 | 34.1% | 100% | ✅ PASS |
| cls_max | 1.00 | 52.3% | 50% | ❌ FAIL |
| mean_mean | 0.90 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.91 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.92 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.93 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.94 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.95 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.96 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.97 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.98 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 0.99 | 67.8% | 50% | ❌ FAIL |
| mean_mean | 1.00 | 67.8% | 50% | ❌ FAIL |
| max_max | 0.90 | 0.0% | 100% | ✅ PASS |
| max_max | 0.91 | 0.0% | 100% | ✅ PASS |
| max_max | 0.92 | 0.0% | 100% | ✅ PASS |
| max_max | 0.93 | 0.0% | 100% | ✅ PASS |
| max_max | 0.94 | 0.0% | 100% | ✅ PASS |
| max_max | 0.95 | 0.0% | 100% | ✅ PASS |
| max_max | 0.96 | 0.0% | 100% | ✅ PASS |
| max_max | 0.97 | 0.0% | 100% | ✅ PASS |
| max_max | 0.98 | 0.0% | 100% | ✅ PASS |
| max_max | 0.99 | 0.0% | 100% | ✅ PASS |
| max_max | 1.00 | 0.0% | 100% | ✅ PASS |

### Scenario: Reasoning (Logic)
| Algo | Multiplier | Savings % | Recall % | Status |
| :--- | :--- | :--- | :--- | :--- |
| mean_max | 0.90 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.91 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.92 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.93 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.94 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.95 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.96 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.97 | 2.7% | 100% | ✅ PASS |
| mean_max | 0.98 | 11.9% | 100% | ✅ PASS |
| mean_max | 0.99 | 29.8% | 86% | ❌ FAIL |
| mean_max | 1.00 | 47.4% | 86% | ❌ FAIL |
| cls_max | 0.90 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.91 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.92 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.93 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.94 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.95 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.96 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.97 | 1.0% | 86% | ❌ FAIL |
| cls_max | 0.98 | 5.3% | 43% | ❌ FAIL |
| cls_max | 0.99 | 14.9% | 29% | ❌ FAIL |
| cls_max | 1.00 | 50.2% | 14% | ❌ FAIL |
| mean_mean | 0.90 | 7.5% | 29% | ❌ FAIL |
| mean_mean | 0.91 | 7.5% | 29% | ❌ FAIL |
| mean_mean | 0.92 | 7.5% | 29% | ❌ FAIL |
| mean_mean | 0.93 | 16.3% | 14% | ❌ FAIL |
| mean_mean | 0.94 | 16.3% | 14% | ❌ FAIL |
| mean_mean | 0.95 | 24.1% | 14% | ❌ FAIL |
| mean_mean | 0.96 | 39.6% | 14% | ❌ FAIL |
| mean_mean | 0.97 | 54.2% | 14% | ❌ FAIL |
| mean_mean | 0.98 | 54.2% | 14% | ❌ FAIL |
| mean_mean | 0.99 | 54.2% | 14% | ❌ FAIL |
| mean_mean | 1.00 | 54.2% | 14% | ❌ FAIL |
| max_max | 0.90 | 0.0% | 100% | ✅ PASS |
| max_max | 0.91 | 0.0% | 100% | ✅ PASS |
| max_max | 0.92 | 0.0% | 100% | ✅ PASS |
| max_max | 0.93 | 0.0% | 100% | ✅ PASS |
| max_max | 0.94 | 0.0% | 100% | ✅ PASS |
| max_max | 0.95 | 0.0% | 100% | ✅ PASS |
| max_max | 0.96 | 0.0% | 100% | ✅ PASS |
| max_max | 0.97 | 0.0% | 100% | ✅ PASS |
| max_max | 0.98 | 0.0% | 100% | ✅ PASS |
| max_max | 0.99 | 0.0% | 100% | ✅ PASS |
| max_max | 1.00 | 0.0% | 100% | ✅ PASS |

### Scenario: Summary (Recap)
| Algo | Multiplier | Savings % | Recall % | Status |
| :--- | :--- | :--- | :--- | :--- |
| mean_max | 0.90 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.91 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.92 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.93 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.94 | 0.0% | 100% | ✅ PASS |
| mean_max | 0.95 | 15.7% | 88% | ✅ PASS |
| mean_max | 0.96 | 18.0% | 75% | ✅ PASS |
| mean_max | 0.97 | 23.6% | 50% | ❌ FAIL |
| mean_max | 0.98 | 36.8% | 38% | ❌ FAIL |
| mean_max | 0.99 | 40.7% | 12% | ❌ FAIL |
| mean_max | 1.00 | 51.9% | 0% | ❌ FAIL |
| cls_max | 0.90 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.91 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.92 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.93 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.94 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.95 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.96 | 0.0% | 100% | ✅ PASS |
| cls_max | 0.97 | 2.9% | 100% | ✅ PASS |
| cls_max | 0.98 | 12.7% | 100% | ✅ PASS |
| cls_max | 0.99 | 26.9% | 100% | ✅ PASS |
| cls_max | 1.00 | 40.6% | 88% | ✅ PASS |
| mean_mean | 0.90 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.91 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.92 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.93 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.94 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.95 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.96 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.97 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.98 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 0.99 | 60.7% | 0% | ❌ FAIL |
| mean_mean | 1.00 | 60.7% | 0% | ❌ FAIL |
| max_max | 0.90 | 0.0% | 100% | ✅ PASS |
| max_max | 0.91 | 0.0% | 100% | ✅ PASS |
| max_max | 0.92 | 0.0% | 100% | ✅ PASS |
| max_max | 0.93 | 0.0% | 100% | ✅ PASS |
| max_max | 0.94 | 0.0% | 100% | ✅ PASS |
| max_max | 0.95 | 0.0% | 100% | ✅ PASS |
| max_max | 0.96 | 0.0% | 100% | ✅ PASS |
| max_max | 0.97 | 0.0% | 100% | ✅ PASS |
| max_max | 0.98 | 0.0% | 100% | ✅ PASS |
| max_max | 0.99 | 0.0% | 100% | ✅ PASS |
| max_max | 1.00 | 0.0% | 100% | ✅ PASS |

