import numpy as np
from src.wamp.core.filter import WAMPruner


def research_attention_resonator():
    print("=== RESEARCH: ATTENTION RESONATOR ROUTING ===")
    pruner = WAMPruner(model_dir="./deberta_small_model")

    # Task descriptions that act as "Resonators"
    resonators = {
        "Fact": "Retrieve a specific pinpoint value, parameter, secret or fact from the context.",
        "Logic": "Compare, analyze, cross-reference or synthesize logic from multiple sources.",
        "Summary": "Provide a broad overview, summary, or condensed version of the entire history.",
    }

    test_queries = {
        "Fact": "What is the database port?",
        "Logic": "Compare projects Alpha and Beta. Who has more budget?",
        "Summary": "Give me a brief summary of our architecture discussion.",
    }

    print(f"{'Actual Task':<10} | {'Resonator':<10} | {'Mean Cross-Attention'}")
    print("-" * 50)

    for q_name, query in test_queries.items():
        query_enc = pruner.tokenizer.encode(query).ids

        for r_name, r_text in resonators.items():
            r_enc = pruner.tokenizer.encode(r_text).ids

            # Concatenate [Resonator] + [Query]
            ids = np.array([r_enc + query_enc], dtype=np.int64)
            mask = np.ones_like(ids)

            outputs = pruner.session.run(
                pruner.output_names, {"input_ids": ids, "attention_mask": mask}
            )

            # Measure attention FROM Resonator tokens TO Query tokens
            # last layer, all heads
            r_len = len(r_enc)
            len(query_enc)

            last_layer = outputs[-1][0]  # (heads, seq, seq)
            # Cross-attention slice: Query rows, Resonator columns (or vice versa)
            # We want to see how much the Query attends to the Resonator
            cross_attn = last_layer[:, r_len:, :r_len]

            resonator_score = np.mean(cross_attn)

            print(f"{q_name:<10} | {r_name:<10} | {resonator_score:.6f}")


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    research_attention_resonator()
