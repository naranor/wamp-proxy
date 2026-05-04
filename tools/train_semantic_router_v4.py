import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import LogisticRegression
from scipy.stats import kurtosis
import logging
import os

def load_dataset(filename):
    """Load DATA list from a python file manually."""
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
    delta_emb = cls_emb - mean_emb
    delta_norm = np.linalg.norm(delta_emb)
    
    # 2. Attention Statistics
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    attn_std = np.std(flat_attn)
    
    # 3. Structural/Heuristic Features
    q_len = len(encoding.ids)
    has_code = 1.0 if any(c in text for c in ["{", "}", "(", ")", "[", "]", "=", ">", "<"]) else 0.0
    logic_words = ["compare", "analyze", "why", "difference", "explain", "evaluate", "contrast", "rewrite", "debug"]
    logic_hits = sum(1 for w in logic_words if w in text.lower()) / len(logic_words)

    return np.concatenate([
        cls_emb, 
        mean_emb, 
        delta_emb, 
        [delta_norm, attn_kurt, attn_max, attn_std, q_len, has_code, logic_hits]
    ])

def train_semantic_router_v4():
    print("=== TRAINING UNIFIED OVR SEMANTIC ROUTER (V4) ===")
    pruner = WAMPruner(model_dir="./model")
    
    # Load all datasets
    summary_tasks = load_dataset("tools/dataset_task_summary.py")
    needle_tasks = load_dataset("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset("tools/dataset_task_reasoning.py")
    
    X_all = []
    y_raw = [] # 0: Summary, 1: Needle, 2: Reasoning
    
    print("Extracting features for all datasets (~300 samples)...")
    for q in summary_tasks: 
        X_all.append(get_enhanced_features(pruner, q))
        y_raw.append(0)
    for q in needle_tasks: 
        X_all.append(get_enhanced_features(pruner, q))
        y_raw.append(1)
    for q in reasoning_tasks: 
        X_all.append(get_enhanced_features(pruner, q))
        y_raw.append(2)
            
    X_all = np.array(X_all)
    y_raw = np.array(y_raw)

    # --- 1. Train NEEDLE Detector (Needle vs All) ---
    print("\nTraining NEEDLE detector...")
    y_needle = (y_raw == 1).astype(int)
    clf_needle = LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5, class_weight='balanced')
    clf_needle.fit(X_all, y_needle)
    print(f"✅ Needle Detector Accuracy: {clf_needle.score(X_all, y_needle)*100:.1f}%")

    # --- 2. Train REASONING Detector (Reasoning vs All) ---
    print("Training REASONING detector...")
    y_reasoning = (y_raw == 2).astype(int)
    clf_reasoning = LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5, class_weight='balanced')
    clf_reasoning.fit(X_all, y_reasoning)
    print(f"✅ Reasoning Detector Accuracy: {clf_reasoning.score(X_all, y_reasoning)*100:.1f}%")
    
    # Save combined weights
    weights = {
        "needle": {
            "coef": clf_needle.coef_.tolist(),
            "intercept": clf_needle.intercept_.tolist()
        },
        "reasoning": {
            "coef": clf_reasoning.coef_.tolist(),
            "intercept": clf_reasoning.intercept_.tolist()
        },
        "model_info": "Unified OVR Semantic Router v4",
        "features": "Composite 2311d [CLS, Mean, Delta, Stats, Heuristics]"
    }
    
    with open("src/wamp/core/router_weights_semantic_v4.json", "w") as f:
        json.dump(weights, f)
    
    print("\n🚀 Enhanced OVR Router weights saved to src/wamp/core/router_weights_semantic_v4.json")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_semantic_router_v4()
