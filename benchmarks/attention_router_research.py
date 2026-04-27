import numpy as np
from src.wamp.core.filter import WAMPruner
from scipy.stats import kurtosis, entropy


def research_attention_routing():
    print("=== RESEARCH: ATTENTION-BASED TASK ROUTING ===")
    pruner = WAMPruner(model_dir="./deberta_small_model")

    test_queries = {
        "Needle": [
            "What is the database port?",
            "Show me the encryption key.",
            "Find the API secret.",
        ],
        "Summary": [
            "Summarize the entire conversation into a single cohesive paragraph.",
            "Provide a comprehensive overview of all mentioned architectural details.",
            "Write a long and detailed report summarizing our previous discussion.",
        ],
        "Reasoning": [
            "Compare project Alpha and Beta in terms of budget and leadership.",
            "Analyze the root cause of the system failure mentioned earlier.",
            "Which team has more funding according to the information fragments?",
        ],
    }

    print(
        f"{'Category':<10} | {'Mean Attn':<10} | {'Max Attn':<10} | {'Entropy':<10} | {'Kurtosis'}"
    )
    print("-" * 60)

    for category, queries in test_queries.items():
        for query in queries:
            encoding = pruner.tokenizer.encode(query)
            ids = np.array([encoding.ids], dtype=np.int64)
            mask = np.array([encoding.attention_mask], dtype=np.int64)

            # Get self-attention of the query (task)
            # layer_attn shape: (batch, heads, seq_len, seq_len)
            outputs = pruner.session.run(
                pruner.output_names, {"input_ids": ids, "attention_mask": mask}
            )

            # Use the last layer's self-attention
            last_layer = outputs[-1][0]  # (heads, seq, seq)
            # Average across heads
            avg_self_attn = np.mean(last_layer, axis=0)  # (seq, seq)
            # Get attention FROM all tokens TO all tokens in the query
            flat_attn = avg_self_attn.flatten()

            m_val = np.mean(flat_attn)
            max_val = np.max(flat_attn)
            ent_val = entropy(flat_attn + 1e-9)
            kurt_val = kurtosis(flat_attn)

            print(f"{category:<10} | {m_val:.6f} | {max_val:.6f} | {ent_val:.4f} | {kurt_val:.2f}")


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    research_attention_routing()
