import numpy as np
from src.wamp.core.filter import WAMPruner
import json
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
    return np.concatenate([cls_emb, mean_emb, [attn_kurt, attn_max, q_len]]), ids[0]

def test_v2_needle_immunity():
    print("=== HYPOTHESIS TEST: REASONING V2 (NEEDLE IMMUNE) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    
    test_suite = {
        "Needle": [
            "What is the port?", "Show me the secret key.", "What is the backup secret?",
            "API password?", "Find server IP.", "DB connection string.", "JWT expiry?",
            "What's the admin pass?", "Find lead scientist name.", "Budget of project Beta?"
        ],
        "Reasoning": [
            "Compare project Alpha and Beta in terms of budget.",
            "Analyze the differences between the current and future pipelines.",
            "Explain the link between Kong and OAuth2."
        ]
    }

    # Load V2 weights
    with open("src/wamp/core/router_weights_v2.json", "r") as f:
        rw = json.load(f)

    print(f"{'Query':<35} | {'p(Reasoning) V2':<15} | {'Status'}")
    print("-" * 65)

    for category, queries in test_suite.items():
        for query in queries:
            features, _ = get_composite_features(pruner, query)
            
            w_r = np.array(rw["coef"])
            b_r = np.array(rw["intercept"])
            score_r = np.dot(w_r, features) + b_r
            prob_r = 1 / (1 + np.exp(-score_r[0]))
            
            # Prediction: If > 0.5 then Reasoning
            predicted = "Reasoning" if prob_r > 0.5 else "Other"
            actual = "Reasoning" if category == "Reasoning" else "Other"
            
            success = (predicted == actual)
            print(f"{query[:35]:<35} | {prob_r:.4f}          | {'✅' if success else '❌'}")

if __name__ == "__main__":
    import logging
    logging.getLogger("wamp").setLevel(logging.ERROR)
    test_v2_needle_immunity()
