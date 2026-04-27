import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
import logging


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
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, q_len]])


def train_reasoning_v2():
    print("=== TRAINING: REASONING V2 (NEEDLE IMMUNE) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    # Class 1: Pure Reasoning
    reasoning = [
        "Compare project Alpha and Beta.",
        "Analyze the system architecture.",
        "Why was the database slow?",
        "Evaluate the security protocols.",
        "Contrast the two scientists.",
        "Synthesize the budget data.",
        "Logic check of the deployment.",
        "Cross-reference logs and events.",
        "Which is more cost-effective?",
        "Explain the link between auth and cache.",
        "How does Kubernetes handle failover?",
        "Verify the integrity.",
        "Relationship between Kong and JWT.",
        "Analyze root cause.",
        "Justify the use of Julia.",
        "Compare budgets 2021 vs 2022.",
        "Critique the design.",
        "Evaluate performance.",
    ] * 5

    # Class 0: Other (Includes Needle to train immunity)
    others = [
        # Summary samples
        "TL;DR of the chat.",
        "Summarize our talk.",
        "Give a brief overview.",
        "Provide a concise report.",
        "Short synopsis.",
        "High-level summary.",
        "Condensed version.",
        "One paragraph summary please.",
        "Recap points.",
        # Needle samples (The Immunity Boost)
        "What is the port?",
        "API key?",
        "DB secret.",
        "JWT expiry.",
        "Admin pass?",
        "Show me the secret key.",
        "What is the backup secret?",
        "Find server IP.",
        "DB connection string.",
        "What's the admin pass?",
        "Find lead scientist name.",
    ] * 5

    X, y = [], []
    print("Extracting V2 composite features...")
    for q in reasoning:
        X.append(get_composite_features(pruner, q))
        y.append(1)
    for q in others:
        X.append(get_composite_features(pruner, q))
        y.append(0)

    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(
        np.array(X), np.array(y)
    )
    print(f"✅ V2 Accuracy (Train): {clf.score(X, y) * 100:.1f}%")

    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "label_map": {"Other": 0, "Reasoning": 1},
        "model_info": "Reasoning V2: Trained to ignore Needle queries",
    }

    with open("src/wamp/core/router_weights_v2.json", "w") as f:
        json.dump(weights, f)

    print("🚀 Reasoning V2 weights saved to src/wamp/core/router_weights_v2.json")


if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_reasoning_v2()
