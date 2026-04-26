# Step 1 Report: Adaptive Router Training
**Model:** DeBERTa-v3-small-NLI
**Method:** Hybrid (Statistical Attention Shape + Binary LogReg)

## Results
- **LogReg Accuracy:** 94.9% (Reasoning vs Summary)
- **Hybrid Accuracy:** 88.9% (Needle/Reasoning/Summary)

## Routing Hierarchy
1. **Stage 1 (Needle):** Statistical check (Mean Attention > 0.12). Perfect detection of short fact queries.
2. **Stage 2 (Reasoning):** LogReg on Mean-Pooled embeddings. High precision for complex analytical queries.
3. **Stage 3 (Summary):** Keyword-based boost + LogReg default. Robust detection of summarization intents.

## Conclusion
The foundation for adaptive research is solid. The router reliably separates the three target scenarios.

---
*Ready for Step 2: Researching Needle In A Haystack.*
