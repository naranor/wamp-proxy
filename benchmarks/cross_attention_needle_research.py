import numpy as np
from src.wamp.core.filter import WAMPruner


def research_cross_attention_needle():
    print("=== RESEARCH: CROSS-ATTENTION NEEDLE DETECTION ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    # These acts as "Magnetic Poles" for fact-retrieval intents
    references = [
        "The specific value or parameter mentioned is 12345.",
        "The secret key or password is agreement.",
        "The system setting for port is 9999.",
    ]

    test_queries = {
        "Needle (Positive)": [
            "What is the port?",
            "Show me the secret key.",
            "Find the DB connection string.",
        ],
        "Reasoning (Negative)": [
            "Compare Alpha and Beta.",
            "Analyze why the system crashed.",
            "Why was project Gamma cancelled?",
        ],
        "Summary (Negative)": [
            "Summarize the talk.",
            "Give me an overview.",
            "TL;DR of the session.",
        ],
    }

    print(f"{'Query Type':<20} | {'Query Text':<30} | {'Resonance Score'}")
    print("-" * 75)

    for q_type, queries in test_queries.items():
        for query in queries:
            q_enc = pruner.tokenizer.encode(query).ids
            q_len = len(q_enc)

            total_resonance = 0.0

            for ref in references:
                ref_enc = pruner.tokenizer.encode(ref).ids
                len(ref_enc)

                # Input: [QUERY] [SEP] [REF]
                ids = np.array([q_enc + ref_enc], dtype=np.int64)
                mask = np.ones_like(ids)

                outputs = pruner.session.run(
                    pruner.output_names, {"input_ids": ids, "attention_mask": mask}
                )

                # Get attention FROM Query TO Reference
                # Last layer, all heads
                last_layer = outputs[-1][0]  # (heads, seq, seq)
                # Slice: rows = query tokens, cols = reference tokens
                res_slice = last_layer[:, :q_len, q_len:]

                # Max attention from any query word to any reference word
                total_resonance += np.max(res_slice)

            avg_resonance = total_resonance / len(references)
            print(f"{q_type:<20} | {query[:30]:<30} | {avg_resonance:.6f}")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    research_cross_attention_needle()
