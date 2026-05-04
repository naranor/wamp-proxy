import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from sklearn.linear_model import Ridge
from scipy.stats import kurtosis
import logging

def get_features(pruner, query, msg):
    # Combine query and msg to get relationship features
    text = f"QUERY: {query} MESSAGE: {msg}"
    encoding = pruner.tokenizer.encode(text)
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
    
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, len(encoding.ids)]])

def train_regression():
    print("=== TRAINING SEMANTIC REGRESSION MODEL ===")
    pruner = WAMPruner(model_dir="./model") # Using the auto-downloaded model
    
    # Dataset: (Query, Message) -> Relevance Score (0.0 to 1.0)
    data = [
        ("What is the port?", "The database port is 5432.", 1.0),
        ("What is the port?", "We use standard PostgreSQL settings.", 0.6),
        ("What is the port?", "I think the weather is nice today.", 0.0),
        ("What is the port?", "Network configuration is located in /etc/net.", 0.4),
        
        ("Compare budget Alpha and Beta", "Alpha budget is 5M, Beta is 3M.", 1.0),
        ("Compare budget Alpha and Beta", "Project Alpha started in 2021.", 0.5),
        ("Compare budget Alpha and Beta", "Scientists are working hard on energy.", 0.2),
        ("Compare budget Alpha and Beta", "I am a helpful AI assistant.", 0.0),
        
        ("Who is the lead scientist?", "Dr. Elena Rossi leads the team.", 1.0),
        ("Who is the lead scientist?", "The team consists of 5 developers.", 0.3),
        ("Who is the lead scientist?", "The laboratory is based in Singapore.", 0.4),
        ("Who is the lead scientist?", "Please let me know if you need more info.", 0.1)
    ] * 10 # Augment
    
    X, y = [], []
    print("Extracting features for regression...")
    for q, m, score in data:
        X.append(get_features(pruner, q, m))
        y.append(score)
        
    # Ridge regression is great for semantic weights
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    print(f"✅ Training R^2 Score: {model.score(X, y):.4f}")
    
    # Save weights
    weights = {
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "model_info": "Semantic Regression v1: Relevance Scorer"
    }
    
    with open("src/wamp/core/regression_weights.json", "w") as f:
        json.dump(weights, f)
    
    print("🚀 Regression weights saved to src/wamp/core/regression_weights.json")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    train_regression()
