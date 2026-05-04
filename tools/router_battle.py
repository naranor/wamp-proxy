import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis
import logging

def get_v1_features(pruner, text):
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
    return np.concatenate([cls_emb, mean_emb, [kurtosis(flat_attn), np.max(flat_attn), len(encoding.ids)]])

def get_v4_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {'input_ids': ids, 'attention_mask': mask})
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    delta_emb = cls_emb - mean_emb
    last_attn = outputs[-1][0]
    avg_attn = np.mean(last_attn, axis=0)
    flat_attn = avg_attn.flatten()
    q_len = len(encoding.ids)
    has_code = 1.0 if any(c in text for c in ["{", "}", "(", ")", "[", "]", "=", ">", "<"]) else 0.0
    logic_hits = sum(1 for w in ["compare", "analyze", "why", "difference"] if w in text.lower()) / 4
    return np.concatenate([cls_emb, mean_emb, delta_emb, [np.linalg.norm(delta_emb), kurtosis(flat_attn), np.max(flat_attn), np.std(flat_attn), q_len, has_code, logic_hits]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def router_battle():
    print("=== THE ROUTER BATTLE: OLD CASCADE VS NEW OVR SEMANTIC ===")
    pruner = WAMPruner(model_dir="./model")
    
    # Load V1 Weights
    with open("src/wamp/core/router_weights.json", "r") as f: v1_rw = json.load(f)
    with open("src/wamp/core/needle_weights.json", "r") as f: v1_nw = json.load(f)
    
    # Load V4 Weights
    with open("src/wamp/core/router_weights_semantic_v4.json", "r") as f: v4_w = json.load(f)

    test_queries = [
        "What is the port?", # Needle
        "Compare project Alpha and Beta.", # Reasoning
        "Analyze why the build failed.", # Reasoning
        "Give me a summary of our chat.", # Summary
        "What was the database port mentioned earlier?", # Hard Needle
        "Can you rewrite this Python function?", # Reasoning
        "API key?", # Needle
        "TL;DR", # Summary
        "Show me the secret key.", # Needle
        "Recap the main architectural points." # Summary
    ]

    print(f"{'Query':<43} | {'Old Router':<12} | {'New Router (V4)':<12} | {'Verdict'}")
    print("-" * 90)

    for query in test_queries:
        # --- OLD ROUTER LOGIC ---
        f1 = get_v1_features(pruner, query)
        p_r1 = sigmoid(np.dot(v1_rw["coef"], f1) + v1_rw["intercept"])[0]
        if p_r1 > 0.5:
            old_winner = "Reasoning"
        else:
            p_n1 = sigmoid(np.dot(v1_nw["coef"], f1) + v1_nw["intercept"])[0]
            old_winner = "Needle" if p_n1 > 0.5 else "Summary"
            
        # --- NEW ROUTER LOGIC ---
        f4 = get_v4_features(pruner, query)
        p_n4 = sigmoid(np.dot(v4_w["needle"]["coef"], f4) + v4_w["needle"]["intercept"])[0]
        p_r4 = sigmoid(np.dot(v4_w["reasoning"]["coef"], f4) + v4_w["reasoning"]["intercept"])[0]
        
        if p_n4 > 0.6: new_winner = "Needle"
        elif p_r4 > 0.5: new_winner = "Reasoning"
        else: new_winner = "Summary"

        status = "SAME" if old_winner == new_winner else "CHANGED"
        print(f"{query[:41]:<43} | {old_winner:<12} | {new_winner:<12} | {status}")

if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    router_battle()
