import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis


def get_composite_features(pruner, query):
    encoding = pruner.tokenizer.encode(query)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
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


def check_query():
    pruner = WAMPruner(model_dir="./deberta_nli_model")
    query = "What was the database port mentioned earlier?"
    features = get_composite_features(pruner, query)

    with open("src/wamp/core/needle_weights.json", "r") as f:
        nw = json.load(f)
    with open("src/wamp/core/router_weights.json", "r") as f:
        rw = json.load(f)

    p_n = 1 / (1 + np.exp(-(np.dot(np.array(nw["coef"]), features) + nw["intercept"])[0]))
    p_r = 1 / (1 + np.exp(-(np.dot(np.array(rw["coef"]), features) + rw["intercept"])[0]))

    print(f"Query: {query}")
    print(f"p(Needle): {p_n:.4f}")
    print(f"p(Reason): {p_r:.4f}")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    check_query()
