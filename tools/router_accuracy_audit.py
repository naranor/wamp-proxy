import numpy as np
from src.wamp.core.filter import WAMPruner
import json
from scipy.stats import kurtosis
import logging


def load_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        loc = {}
        exec(f.read(), {}, loc)
        return loc.get("DATA", [])


def get_v1_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
    last_hidden = outputs[0][0]
    return np.concatenate(
        [
            last_hidden[0],
            np.mean(last_hidden, axis=0),
            [
                kurtosis(np.mean(outputs[-1][0], axis=0).flatten()),
                np.max(outputs[-1][0]),
                len(encoding.ids),
            ],
        ]
    )


def get_v4_features(pruner, text):
    encoding = pruner.tokenizer.encode(text)
    ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
    last_hidden = outputs[0][0]
    cls_emb = last_hidden[0]
    mean_emb = np.mean(last_hidden, axis=0)
    delta_emb = cls_emb - mean_emb
    flat_attn = np.mean(outputs[-1][0], axis=0).flatten()
    return np.concatenate(
        [
            cls_emb,
            mean_emb,
            delta_emb,
            [
                np.linalg.norm(delta_emb),
                kurtosis(flat_attn),
                np.max(flat_attn),
                np.std(flat_attn),
                len(encoding.ids),
                0.0,
                0.0,
            ],
        ]
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_accuracy_audit():
    print("=== FINAL ACCURACY AUDIT: OLD V1 VS NEW UNIVERSAL V4 ===")
    pruner = WAMPruner(model_dir="./model")

    # Load all datasets
    ds = {
        "Summary": load_dataset("tools/dataset_task_summary.py"),
        "Needle": load_dataset("tools/dataset_task_needle.py"),
        "Reasoning": load_dataset("tools/dataset_task_reasoning.py"),
    }

    # Load Weights
    with open("src/wamp/core/router_weights.json", "r") as f:
        v1_rw = json.load(f)
    with open("src/wamp/core/needle_weights.json", "r") as f:
        v1_nw = json.load(f)
    with open("src/wamp/core/router_weights_semantic_v4.json", "r") as f:
        v4_w = json.load(f)

    stats = {
        "V1": {"correct": 0, "total": 0, "by_cat": {"Summary": 0, "Needle": 0, "Reasoning": 0}},
        "V4": {"correct": 0, "total": 0, "by_cat": {"Summary": 0, "Needle": 0, "Reasoning": 0}},
    }

    for cat_name, queries in ds.items():
        print(f"Auditing category: {cat_name} ({len(queries)} samples)...")
        for q in queries:
            # --- V1 Cascade ---
            f1 = get_v1_features(pruner, q)
            p_r1 = sigmoid(np.dot(v1_rw["coef"], f1) + v1_rw["intercept"])[0]
            v1_pred = (
                "Reasoning"
                if p_r1 > 0.5
                else (
                    "Needle"
                    if sigmoid(np.dot(v1_nw["coef"], f1) + v1_nw["intercept"])[0] > 0.5
                    else "Summary"
                )
            )

            # --- V4 OVR ---
            f4 = get_v4_features(pruner, q)
            p_n4 = sigmoid(np.dot(v4_w["needle"]["coef"], f4) + v4_w["needle"]["intercept"])[0]
            p_r4 = sigmoid(np.dot(v4_w["reasoning"]["coef"], f4) + v4_w["reasoning"]["intercept"])[
                0
            ]
            v4_pred = "Needle" if p_n4 > 0.6 else ("Reasoning" if p_r4 > 0.5 else "Summary")

            if v1_pred == cat_name:
                stats["V1"]["correct"] += 1
                stats["V1"]["by_cat"][cat_name] += 1
            if v4_pred == cat_name:
                stats["V4"]["correct"] += 1
                stats["V4"]["by_cat"][cat_name] += 1

            stats["V1"]["total"] += 1
            stats["V4"]["total"] += 1

    print("\n--- RESULTS TABLE ---")
    print(f"{'Category':<15} | {'Old Router (V1)':<18} | {'New Router (V4)':<18}")
    print("-" * 60)
    for cat in ds.keys():
        v1_cat_acc = stats["V1"]["by_cat"][cat] / len(ds[cat]) * 100
        v4_cat_acc = stats["V4"]["by_cat"][cat] / len(ds[cat]) * 100
        print(f"{cat:<15} | {v1_cat_acc:>6.1f}%            | {v4_cat_acc:>6.1f}%")

    v1_total = stats["V1"]["correct"] / stats["V1"]["total"] * 100
    v4_total = stats["V4"]["correct"] / stats["V4"]["total"] * 100
    print("-" * 60)
    print(f"{'OVERALL':<15} | {v1_total:>6.1f}%            | {v4_total:>6.1f}%")


if __name__ == "__main__":
    logging.getLogger("wamp").setLevel(logging.ERROR)
    run_accuracy_audit()
