import numpy as np
from src.wamp.core.filter import WAMPruner
import pandas as pd


def tune_needle_detector():
    print("=== TUNING NEEDLE DETECTOR (Statistical Thresholds) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    dataset = {
        "Needle": [
            "What is the port?",
            "API key?",
            "Database secret.",
            "JWT expiry.",
            "Admin pass?",
            "What is the port number for the database?",
            "Give me the API key mentioned earlier.",
            "What was the backup encryption password?",
            "Find the lead scientist name.",
            "Identify the server IP from the logs.",
            "Port?",
            "Key?",
            "Secret?",
            "Password?",
            "What is the database port?",
            "Show me the encryption key.",
            "Find the API secret.",
        ],
        "Non-Needle": [
            "Summarize our talk.",
            "Compare Alpha and Beta.",
            "Why did it fail?",
            "Recap the main points.",
            "Explain the architecture.",
            "Analyze the budget.",
            "Give me an overview.",
            "How does Kubernetes work?",
            "Write a report.",
            "TL;DR.",
            "Compare projects.",
            "Synthesize the team structure.",
            "Evaluation of monitoring.",
            "Contrast pipelines.",
            "What is the general topic?",
            "Briefly outline the system.",
            "Technical synopsis.",
        ],
    }

    results = []

    print("Analyzing attention shapes...")
    for label, queries in dataset.items():
        for query in queries:
            encoding = pruner.tokenizer.encode(query)
            ids = np.array([encoding.ids], dtype=np.int64)
            mask = np.array([encoding.attention_mask], dtype=np.int64)

            outputs = pruner.session.run(None, {"input_ids": ids, "attention_mask": mask})
            # Last layer self-attention
            last_attn = outputs[-1][0]

            mean_attn = np.mean(last_attn)
            flat_attn = last_attn.flatten()
            ent_val = -np.sum(flat_attn * np.log(flat_attn + 1e-9)) / len(flat_attn)

            results.append({"label": label, "mean": mean_attn, "entropy": ent_val, "query": query})

    df = pd.DataFrame(results)

    print("\n--- STATISTICS BY CLASS ---")
    print(df.groupby("label")[["mean", "entropy"]].agg(["mean", "std", "min", "max"]))

    # Simple heuristic to find best threshold
    # We want a Mean threshold that captures ALL Needles
    min_needle_mean = df[df["label"] == "Needle"]["mean"].min()
    max_non_needle_mean = df[df["label"] == "Non-Needle"]["mean"].max()

    print(f"\nRecommended Mean Threshold (Min Needle): {min_needle_mean:.4f}")
    print(f"Potential Conflict (Max Non-Needle):    {max_non_needle_mean:.4f}")

    # Same for Entropy
    max_needle_ent = df[df["label"] == "Needle"]["entropy"].max()
    min_non_needle_ent = df[df["label"] == "Non-Needle"]["entropy"].min()

    print(f"\nRecommended Entropy Threshold (Max Needle): {max_needle_ent:.4f}")
    print(f"Potential Conflict (Min Non-Needle):      {min_non_needle_ent:.4f}")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    tune_needle_detector()
