import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis

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

def train_reasoning_vs_all():
    print("=== TRAINING: REASONING vs EVERYTHING ELSE ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    # Class 1: Reasoning (Pure analysis and logic)
    reasoning = [
        "Compare project Alpha and Beta.", "Analyze the system architecture.", "Why was the database slow?",
        "Evaluate the security protocols.", "Contrast the two scientists.", "Synthesize the budget data.",
        "Logic check of the deployment.", "Cross-reference logs and events.", "Which is more cost-effective?",
        "Explain the link between auth and cache.", "How does Kubernetes handle failover?", "Verify the integrity.",
        "What are the benefits of Redis here?", "Relationship between Kong and JWT.", "Analyze root cause.",
        "Justify the use of Julia.", "Compare budgets 2021 vs 2022.", "Critique the microservices design.",
        "Evaluate performance.", "Contrast rolling updates vs blue-green.", "Logic of the cluster.",
        "Analyze encryption algorithm.", "How to improve latency?", "Compare read-replicas count."
    ] * 3 # ~72 examples
    
    # Class 0: Everything Else (Needle + Summary)
    others = [
        # Needle samples
        "What is the port?", "API key?", "DB secret.", "JWT expiry.", "Admin pass?",
        "Give me the encryption key.", "Find the lead scientist.", "What's the password?",
        "Identify server IP.", "Budget Alpha?", "K8s version.", "Auth token.",
        # Summary samples
        "TL;DR of the chat.", "Summarize our talk.", "Give a brief overview.",
        "Provide a concise report.", "Short synopsis.", "High-level summary.",
        "Condensed version.", "One paragraph summary please.", "Recap points.",
        "General topic?", "Short version.", "Summarize everything."
    ] * 3 # ~72 examples
    
    X = []
    y = []
    
    print("Extracting composite features...")
    for q in reasoning:
        X.append(get_composite_features(pruner, q))
        y.append(1) # Reasoning
    
    for q in others:
        X.append(get_composite_features(pruner, q))
        y.append(0) # Other
            
    X = np.array(X)
    y = np.array(y)
    
    clf = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced')
    clf.fit(X, y)
    
    print(f"✅ Reasoning vs All Accuracy: {clf.score(X, y)*100:.1f}%")
    
    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "label_map": {"Other": 0, "Reasoning": 1},
        "model_info": "Binary Classifier: Reasoning (1) vs Everything Else (0)"
    }
    
    with open("src/wamp/core/router_weights.json", "w") as f:
        json.dump(weights, f)
    
    print("🚀 Reasoning-vs-All weights saved to src/wamp/core/router_weights.json")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_reasoning_vs_all()
