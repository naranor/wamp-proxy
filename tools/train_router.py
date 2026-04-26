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
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    
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

def train_all_classifiers():
    print("=== UNIFIED TRAINING: COMPOSITE ADAPTIVE ENGINE ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    # 1. Datasets
    needle_queries = [
        "What is the port?", "API key?", "DB secret.", "JWT expiry.", "Admin pass?",
        "What was the database port?", "Give me the API key.", "Encryption password?",
        "Find the lead scientist.", "Identify the server IP.", "Budget Alpha?", "K8s version.",
        "Password for root?", "Secret token?", "Redis port?", "Retrieve the key.",
        "Dr. Rossi location?", "Alpha budget amount.", "Client secret.", "Replica IP.",
        "Memory limit.", "Julia version.", "Token expiry.", "Node count."
    ] * 5
    
    reasoning_queries = [
        "Compare project Alpha and Beta.", "Analyze the system architecture.", "Why was the database slow?",
        "Evaluate the security protocols.", "Contrast the two scientists.", "Synthesize the budget data.",
        "Logic check of the deployment.", "Cross-reference logs and events.", "Which is more cost-effective?",
        "Explain the link between auth and cache.", "How does Kubernetes handle failover?", "Verify the integrity.",
        "Relationship between Kong and JWT.", "Analyze root cause.", "Justify the use of Julia.",
        "Compare budgets 2021 vs 2022.", "Critique the design.", "Evaluate performance."
    ] * 5
    
    summary_queries = [
        "TL;DR of the chat.", "Summarize our talk.", "Give a brief overview.",
        "Provide a concise report.", "Short synopsis.", "High-level summary.",
        "Condensed version.", "One paragraph summary please.", "Recap the main points.",
        "General topic?", "Short version.", "Summarize everything."
    ] * 5

    # --- Training Needle vs All (Stage 1) ---
    print("\nTraining Stage 1: Needle vs All...")
    X, y = [], []
    for q in needle_queries:
        X.append(get_composite_features(pruner, q)); y.append(1)
    for q in reasoning_queries + summary_queries:
        X.append(get_composite_features(pruner, q)); y.append(0)
    
    clf_n = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced').fit(np.array(X), np.array(y))
    print(f"✅ Needle Accuracy: {clf_n.score(X, y)*100:.1f}%")
    
    with open("src/wamp/core/needle_weights.json", "w") as f:
        json.dump({"coef": clf_n.coef_.tolist(), "intercept": clf_n.intercept_.tolist()}, f)

    # --- Training Reasoning vs All (Stage 2) ---
    print("\nTraining Stage 2: Reasoning vs All...")
    X, y = [], []
    for q in reasoning_queries:
        X.append(get_composite_features(pruner, q)); y.append(1)
    for q in needle_queries + summary_queries:
        X.append(get_composite_features(pruner, q)); y.append(0)
    
    clf_r = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced').fit(np.array(X), np.array(y))
    print(f"✅ Reasoning Accuracy: {clf_r.score(X, y)*100:.1f}%")
    
    with open("src/wamp/core/router_weights.json", "w") as f:
        json.dump({"coef": clf_r.coef_.tolist(), "intercept": clf_r.intercept_.tolist()}, f)

    print("\n🚀 All composite weights saved to src/wamp/core/")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_all_classifiers()
