import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis


def get_composite_features(pruner, query):
    encoding = pruner.tokenizer.encode(query)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    q_len = len(encoding.ids)
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, q_len]]), ids[0]


def test_hypothesis_priority_needle():
    print("=== HYPOTHESIS TEST: PRIORITY NEEDLE (OVERRIDE) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    # We focus on the problematic queries from the previous audit
    test_suite = {
        "Needle": [
            "What is the port?",
            "Show me the secret key.",
            "What is the backup secret?",
            "API password?",
            "Find server IP.",
            "DB connection string.",
            "JWT expiry?",
            "What's the admin pass?",
            "Find lead scientist name.",
            "Budget of project Beta?",
        ],
        "Reasoning": [
            "Compare project Alpha and Beta in terms of budget.",
            "Analyze the differences between the current and future pipelines.",
            "Explain the link between Kong and OAuth2.",
        ],
    }

    # Load weights
    with open("src/wamp/core/needle_weights.json", "r") as f:
        nw = json.load(f)
    with open("src/wamp/core/router_weights.json", "r") as f:
        rw = json.load(f)

    print(f"{'Query':<35} | {'p(Needle)':<8} | {'p(Reason)':<8} | {'Result':<10} | {'Status'}")
    print("-" * 85)

    for category, queries in test_suite.items():
        for query in queries:
            features, _ = get_composite_features(pruner, query)

            # 1. Calc probabilities
            p_needle = 1 / (
                1 + np.exp(-(np.dot(np.array(nw["coef"]), features) + nw["intercept"])[0])
            )
            p_reason = 1 / (
                1 + np.exp(-(np.dot(np.array(rw["coef"]), features) + rw["intercept"])[0])
            )

            # --- HYPOTHESIS LOGIC ---
            if p_needle > 0.7:
                predicted = "Needle"
            elif p_reason > 0.5:
                predicted = "Reasoning"
            else:
                predicted = "Summary"

            success = predicted == category
            print(
                f"{query[:35]:<35} | {p_needle:.4f} | {p_reason:.4f} | {predicted:<10} | {'✅' if success else '❌'}"
            )


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    test_hypothesis_priority_needle()
