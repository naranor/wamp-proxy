import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis
import logging

def get_task_features(pruner, text):
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

def audit_semantic_router():
    print("=== AUDIT: UNIFIED SEMANTIC ROUTER (3-CLASS PROBABILITIES) ===")
    pruner = WAMPruner(model_dir="./model")
    
    # Load semantic weights
    with open("src/wamp/core/router_weights_semantic.json", "r") as f:
        weights = json.load(f)
    
    coef = np.array(weights["coef"])
    intercept = np.array(weights["intercept"])
    label_map = weights["label_map"]
    inv_map = {v: k for k, v in label_map.items()}

    test_queries = [
        "What is the database port?", # Expected: Needle
        "Compare Alpha and Beta projects.", # Expected: Reasoning
        "Give me a summary of the chat.", # Expected: Summary
        "Analyze why the build failed.", # Expected: Reasoning
        "API key?", # Expected: Needle
        "TL;DR", # Expected: Summary
        "What was the database port mentioned earlier?", # Hard case
        "Can you rewrite this Python function?", # Reasoning
    ]

    print(f"{'Query':<45} | {'Summary':<8} | {'Needle':<8} | {'Reason':<8} | {'Winner'}")
    print("-" * 90)

    for query in test_queries:
        features = get_task_features(pruner, query)
        
        # Softmax for multinomial logistic regression
        scores = np.dot(coef, features) + intercept
        exp_scores = np.exp(scores - np.max(scores)) # Stability trick
        probs = exp_scores / exp_scores.sum()
        
        winner_idx = np.argmax(probs)
        winner_label = inv_map[winner_idx]
        
        print(f"{query[:43]:<45} | {probs[0]:.3f}   | {probs[1]:.3f}   | {probs[2]:.3f}   | {winner_label}")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    audit_semantic_router()
