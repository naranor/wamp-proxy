from setfit import SetFitModel
import numpy as np
import os
import logging


def load_dataset_custom(filename):
    with open(filename, "r", encoding="utf-8") as f:
        loc = {}
        exec(f.read(), {}, loc)
        return loc.get("DATA", [])


def audit_setfit():
    print("=== FINAL ACCURACY AUDIT: SETFIT SEMANTIC ROUTER ===")

    # 1. Load Model
    model_path = "./setfit_router_model"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = SetFitModel.from_pretrained(model_path)

    # 2. Load all datasets
    ds = {
        "Summary": load_dataset_custom("tools/dataset_task_summary.py"),
        "Needle": load_dataset_custom("tools/dataset_task_needle.py"),
        "Reasoning": load_dataset_custom("tools/dataset_task_reasoning.py"),
    }

    stats = {"correct": 0, "total": 0, "by_cat": {"Summary": 0, "Needle": 0, "Reasoning": 0}}
    inv_map = {0: "Summary", 1: "Needle", 2: "Reasoning"}

    for cat_name, queries in ds.items():
        print(f"Auditing category: {cat_name} ({len(queries)} samples)...")
        for q in queries:
            # SetFit prediction
            probs = model.predict_proba([q])[0]
            winner_idx = np.argmax(probs)
            pred_label = inv_map[int(winner_idx)]

            if pred_label == cat_name:
                stats["correct"] += 1
                stats["by_cat"][cat_name] += 1

            stats["total"] += 1

    print("\n--- SETFIT ACCURACY RESULTS ---")
    print(f"{'Category':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for cat, queries in ds.items():
        acc = stats["by_cat"][cat] / len(queries) * 100
        print(f"{cat:<15} | {acc:>6.1f}%")

    total_acc = stats["correct"] / stats["total"] * 100
    print("-" * 30)
    print(f"{'OVERALL':<15} | {total_acc:>6.1f}%")

    # Sample check for regression probabilities
    print("\n--- SAMPLE SEMANTIC REGRESSION (PROBS) ---")
    test_q = "What was the database port mentioned earlier?"
    p = model.predict_proba([test_q])[0]
    print(f"Query: {test_q}")
    print(f"Summary: {p[0]:.4f} | Needle: {p[1]:.4f} | Reason: {p[2]:.4f}")


if __name__ == "__main__":
    logging.getLogger("setfit").setLevel(logging.ERROR)
    audit_setfit()
