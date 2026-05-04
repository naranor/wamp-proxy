import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis
import logging

def get_enhanced_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    delta_emb = cls_emb - mean_emb
    delta_norm = np.linalg.norm(delta_emb)
    
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    attn_kurt = kurtosis(flat_attn)
    attn_max = np.max(flat_attn)
    attn_std = np.std(flat_attn)
    
    q_len = len(encoding.ids)
    has_code = 1.0 if any(c in text for c in ["{", "}", "(", ")", "[", "]", "=", ">", "<"]) else 0.0
    logic_words = ["compare", "analyze", "why", "difference", "explain", "evaluate", "contrast", "rewrite", "debug"]
    logic_hits = sum(1 for w in logic_words if w in text.lower()) / len(logic_words)

    return np.concatenate([cls_emb, mean_emb, delta_emb, [delta_norm, attn_kurt, attn_max, attn_std, q_len, has_code, logic_hits]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def audit_semantic_router_v4():
    print("=== AUDIT: OVR SEMANTIC ROUTER V4 (90%+ ACCURACY) ===")
    pruner = WAMPruner(model_dir="./model")
    
    with open("src/wamp/core/router_weights_semantic_v4.json", "r") as f:
        weights = json.load(f)
    
    # Needle model
    w_n = np.array(weights["needle"]["coef"])
    b_n = np.array(weights["needle"]["intercept"])
    
    # Reasoning model
    w_r = np.array(weights["reasoning"]["coef"])
    b_r = np.array(weights["reasoning"]["intercept"])

    test_queries = [
        "What is the database port?", # Expected: Needle
        "Compare Alpha and Beta projects.", # Expected: Reasoning
        "Give me a summary of the chat.", # Expected: Summary
        "Analyze why the build failed.", # Expected: Reasoning
        "API key?", # Expected: Needle
        "TL;DR", # Expected: Summary
        "Can you rewrite this Python function?", # Reasoning
        "What was the database port mentioned earlier?", # Needle
    ]

    print(f"{'Query':<45} | {'p(Needle)':<10} | {'p(Reason)':<10} | {'Winner'}")
    print("-" * 85)

    for query in test_queries:
        features = get_enhanced_features(pruner, query)
        
        prob_n = sigmoid(np.dot(w_n, features) + b_n)[0]
        prob_r = sigmoid(np.dot(w_r, features) + b_r)[0]
        
        if prob_n > 0.6: # Needle threshold
            winner = "Needle"
        elif prob_r > 0.5: # Reasoning threshold
            winner = "Reasoning"
        else:
            winner = "Summary"
        
        print(f"{query[:43]:<45} | {prob_n:.4f}     | {prob_r:.4f}     | {winner}")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    audit_semantic_router_v4()
