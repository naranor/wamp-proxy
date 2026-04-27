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


def audit_stage1_composite():
    print("=== AUDIT: COMPOSITE STAGE 1 NEEDLE DETECTION ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    test_suite = {
        "Needle (Positive)": [
            "What is the port?",
            "Show me the secret key.",
            "What's the password?",
            "Find the DB connection string.",
            "API token?",
            "Backup encryption key?",
            "Identify the server IP.",
            "Port number for the cluster?",
            "JWT secret?",
            "What is the value of the constant?",
        ],
        "Other (Negative)": [
            "Summarize the chat.",
            "Compare Alpha and Beta.",
            "Analyze the logs.",
            "Write a concise report.",
            "Why was Gamma cancelled?",
            "Explain the auth flow.",
            "How does Kubernetes work?",
            "Give me an overview.",
            "TL;DR of the session.",
            "Synthesize the project status.",
        ],
    }

    correct = 0
    total = 0

    print(f"{'Test Query':<40} | {'Prediction':<10} | {'Prob':<6} | {'Status'}")
    print("-" * 75)

    # Load weights
    with open("src/wamp/core/needle_weights.json", "r") as f:
        needle_weights = json.load(f)
    w = np.array(needle_weights["coef"])
    b = np.array(needle_weights["intercept"])

    for category, queries in test_suite.items():
        for query in queries:
            features, ids = get_composite_features(pruner, query)

            # Semantic Prob
            score = np.dot(w, features) + b
            prob = 1 / (1 + np.exp(-score[0]))

            # Prediction
            is_needle = prob > 0.5

            label = "Needle" if is_needle else "Other"
            actual = "Needle" if "Positive" in category else "Other"

            success = label == actual
            if success:
                correct += 1
            total += 1

            print(f"{query[:40]:<40} | {label:<10} | {prob:.2f} | {'✅' if success else '❌'}")

    print(f"\nFinal Composite Accuracy: {correct / total * 100:.1f}%")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    audit_stage1_composite()
