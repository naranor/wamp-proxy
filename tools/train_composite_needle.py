import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis


def get_composite_features(pruner, query):
    encoding = pruner.tokenizer.encode(query)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)

    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})

    # 1. Last hidden state features
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)

    # 2. Attention shape features (last layer)
    last_attn = outputs[-1][0]  # (heads, seq, seq)
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()

    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)

    # 3. Structural features
    q_len = len(encoding.ids)

    # Concatenate all into one vector
    # [CLS (hidden), Mean (hidden), Kurtosis (1), Max (1), Length (1)]
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, q_len]])


def train_composite_needle_classifier():
    print("=== TRAINING COMPOSITE NEEDLE CLASSIFIER ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    # Inlined Dataset
    data = {
        "Needle": [
            "What is the port?",
            "API key?",
            "DB secret.",
            "JWT expiry.",
            "Admin pass?",
            "What was the database port mentioned earlier?",
            "Give me the API key mentioned earlier.",
            "What was the backup encryption password?",
            "Find the lead scientist name in the fragments.",
            "Identify the server IP address from the previous fragments.",
            "What is the budget allocation for project Beta?",
            "Tell me the cancellation date of project Gamma.",
            "Password for root access?",
            "What is the specific version of Kubernetes?",
            "Look for the security protocol name.",
            "Retrieve the private key.",
            "Find the database connection string.",
            "Identify the firewall status for segment 3.",
            "What is the name of the ELK stack coordinator?",
            "Give me the OAuth2 client secret.",
            "What is the salt for password hashing?",
            "Find the IP of the read-replica.",
            "What is the memory limit for the cluster?",
            "Tell me the name of the scientist in Singapore.",
            "What is the Julia version used for Beta?",
            "Find the project Alpha budget amount.",
            "Identify the CyberShield department head.",
            "What is the token expiration period?",
            "Find the Elasticsearch node count.",
            "Give me the value of the 'WAMP' constant.",
        ]
        * 4,
        "Other": [
            "Summarize project Alpha goals.",
            "Compare the two database setups.",
            "Why was project Gamma cancelled?",
            "Analyze the system bottlenecks.",
            "Evaluate the monitoring stack.",
            "Contrast GitLab and GitHub CI.",
            "Give me a high-level overview of the tech stack.",
            "TL;DR of the whole chat.",
            "Briefly recap the auth flow.",
            "Write a concise report based on all fragments.",
            "Summarize the entire infrastructure discussion.",
            "Analyze why the system crashed.",
            "Explain the link between auth and cache.",
            "Verify the integrity of the spec.",
            "Is project Alpha more funded than Beta?",
            "Synthesize the team structure.",
            "Contrast monolith and microservices.",
            "Provide a technical synopsis of the meeting.",
        ]
        * 4,
    }

    X = []
    y = []

    print("Extracting composite features...")
    for label, queries in data.items():
        print(f"  Processing {label}...")
        for q in queries:
            X.append(get_composite_features(pruner, q))
            y.append(1 if label == "Needle" else 0)

    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    clf.fit(X, y)

    print(f"✅ Composite Training Accuracy: {clf.score(X, y) * 100:.1f}%")

    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "model_info": "Composite Needle Detector (CLS + Mean + Stats + Length)",
    }

    with open("src/wamp/core/needle_weights.json", "w") as f:
        json.dump(weights, f)

    print("🚀 Composite Needle weights saved.")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_composite_needle_classifier()
