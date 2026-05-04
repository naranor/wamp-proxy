import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
import logging
import os

def load_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        loc = {}
        exec(f.read(), {}, loc)
        return loc.get("DATA", [])

def get_enhanced_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    
    # 1. Semantic Embeddings
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    
    # NEW: Semantic Delta (Captures focus shifts)
    delta_emb = cls_emb - mean_emb
    delta_norm = np.linalg.norm(delta_emb)
    
    # 2. Attention Statistics
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    attn_std = np.std(flat_attn) # NEW: Dispersion of attention
    
    # 3. Structural/Heuristic Features
    q_len = len(encoding.ids)
    has_code = 1.0 if any(c in text for c in ["{", "}", "(", ")", "[", "]", "=", ">", "<"]) else 0.0
    
    # Logic keywords boost
    logic_words = ["compare", "analyze", "why", "difference", "explain", "evaluate", "contrast", "rewrite", "debug"]
    logic_hits = sum(1 for w in logic_words if w in text.lower()) / len(logic_words)

    return np.concatenate([
        cls_emb, 
        mean_emb, 
        delta_emb, 
        [delta_norm, attn_kurt, attn_max, attn_std, q_len, has_code, logic_hits]
    ])

def train_semantic_router_v3():
    print("=== TRAINING ENHANCED SEMANTIC ROUTER (V3) ===")
    pruner = WAMPruner(model_dir="./model")
    
    # Load data
    summary_tasks = load_dataset("tools/dataset_task_summary.py")
    needle_tasks = load_dataset("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset("tools/dataset_task_reasoning.py")
    
    X, y = [], []
    
    print("Extracting enhanced features...")
    for q in summary_tasks: X.append(get_enhanced_features(pruner, q)); y.append(0)
    for q in needle_tasks: X.append(get_enhanced_features(pruner, q)); y.append(1)
    for q in reasoning_tasks: X.append(get_enhanced_features(pruner, q)); y.append(2)
            
    # Multinomial Logistic Regression with stronger penalty
    clf = LogisticRegression(
        solver='lbfgs', 
        max_iter=3000, 
        C=0.1, # Stronger regularization for the complex feature set
        class_weight='balanced'
    )
    
    X = np.array(X)
    y = np.array(y)
    clf.fit(X, y)
    
    print(f"\n✅ V3 Training Accuracy: {clf.score(X, y)*100:.1f}%")
    
    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "label_map": {"Summary": 0, "Needle": 1, "Reasoning": 2},
        "model_info": "Unified Semantic Router v3 (Enhanced Features)",
        "features": "Composite 2311d [CLS, Mean, Delta, Stats, Heuristics]"
    }
    
    with open("src/wamp/core/router_weights_semantic.json", "w") as f:
        json.dump(weights, f)
    
    print("🚀 Enhanced Semantic Router weights saved.")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_semantic_router_v3()
