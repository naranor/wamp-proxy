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
        content = f.read()
        loc = {}
        exec(content, {}, loc)
        return loc.get("DATA", [])

def get_task_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    
    # Composite Features (consistent with our research)
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, len(encoding.ids)]])

def train_semantic_router():
    print("=== TRAINING UNIFIED SEMANTIC ROUTER (V2) ===")
    # Using the currently configured model in ./model as requested
    pruner = WAMPruner(model_dir="./model")
    
    # Load data
    summary_tasks = load_dataset("tools/dataset_task_summary.py")
    needle_tasks = load_dataset("tools/dataset_task_needle.py")
    reasoning_tasks = load_dataset("tools/dataset_task_reasoning.py")
    
    # Prepare data for 3 classes: 0: Summary, 1: Needle, 2: Reasoning
    X, y = [], []
    
    print("Extracting features for Summary tasks...")
    for q in summary_tasks:
        X.append(get_task_features(pruner, q))
        y.append(0)
        
    print("Extracting features for Needle tasks...")
    for q in needle_tasks:
        X.append(get_task_features(pruner, q))
        y.append(1)
        
    print("Extracting features for Reasoning tasks...")
    for q in reasoning_tasks:
        X.append(get_task_features(pruner, q))
        y.append(2)
            
    # Multinomial Logistic Regression (provides continuous semantic scores for all classes)
    clf = LogisticRegression(
        solver='lbfgs', 
        max_iter=2000, 
        C=0.5, # Stronger regularization for better generalization
        class_weight='balanced'
    )
    
    X = np.array(X)
    y = np.array(y)
    clf.fit(X, y)
    
    accuracy = clf.score(X, y) * 100
    print(f"\n✅ Training Accuracy: {accuracy:.1f}%")
    
    # Save weights to a specific file
    weights = {
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "label_map": {"Summary": 0, "Needle": 1, "Reasoning": 2},
        "model_info": "Unified Semantic Router v2 (3-class)",
        "features": "Composite 1539d [CLS, Mean, Kurt, Max, Len]"
    }
    
    with open("src/wamp/core/router_weights_semantic.json", "w") as f:
        json.dump(weights, f)
    
    print("🚀 Semantic Router weights saved to src/wamp/core/router_weights_semantic.json")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_semantic_router()
