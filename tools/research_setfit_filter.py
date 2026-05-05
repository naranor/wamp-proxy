import numpy as np
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_setfit_research():
    print("=== RESEARCH STEP 6: SETFIT AS CONTEXT PRUNER (MINILM-L12) ===")

    # Force use the new SetFit ONNX model
    pruner = WAMPruner(model_dir="./model_setfit_onnx")

    scenarios = {
        "Needle (Fact)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "What was the database port mentioned earlier?",
            "critical": [51, 52],
            "target_recall": 100,
        },
        "Reasoning (Logic)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113],
            "target_recall": 100,
        },
        "Summary (Recap)": {
            "msgs": generate_coherence_long(),
            "task": "Summarize the technical specification of the architecture.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99],
            "target_recall": 75,
        },
    }

    algorithms = ["mean_max", "cls_max", "mean_mean", "max_max"]
    multipliers = np.arange(0.90, 1.01, 0.01)

    results = []

    for name, data in scenarios.items():
        print(f"\n--- Testing Scenario: {name} ---")
        msgs = data["msgs"]
        task_query = data["task"]
        critical = data["critical"]

        # Calculate original tokens (approx)
        total_tokens_before = sum(len(pruner.tokenizer.encode(m["content"]).ids) for m in msgs)

        task_prompt = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{task_query}'"
        task_enc = pruner.tokenizer.encode(task_prompt)

        for algo in algorithms:
            for mult in multipliers:
                # 1. Calculate scores for all messages
                scores_dict = {}
                for i, msg in enumerate(msgs):
                    if i == 0 or i >= len(msgs) - 2:
                        continue  # Skip system/last

                    content = pruner.get_content(msg)
                    msg_enc = pruner.tokenizer.encode(content)

                    # Prepare input pair
                    input_ids = task_enc.ids + msg_enc.ids
                    if len(input_ids) > 1024:
                        input_ids = input_ids[:1024]

                    ids_np = np.array([input_ids], dtype=np.int64)
                    mask_np = np.ones_like(ids_np)

                    outputs = pruner.session.run(
                        None, {"input_ids": ids_np, "attention_mask": mask_np}
                    )

                    task_len, msg_len = len(task_enc.ids), len(msg_enc.ids)
                    # Use last 2 layers
                    layers = outputs[-2:]

                    importance = 0
                    for layer_attn in layers:
                        slc = layer_attn[0, :, :task_len, task_len : task_len + msg_len]
                        if slc.size == 0:
                            continue

                        if algo == "mean_max":
                            importance += float(np.mean(np.max(slc, axis=2)))
                        elif algo == "cls_max":
                            importance += float(np.max(slc[:, 0, :]))
                        elif algo == "mean_mean":
                            importance += float(np.mean(slc))

                    scores_dict[i] = importance / len(layers)

                # 2. Apply threshold
                scores = list(scores_dict.values())
                baseline = np.median(scores) if scores else 0
                threshold = baseline * mult

                filtered = [msgs[0]]  # System
                for i in range(1, len(msgs) - 2):
                    if scores_dict.get(i, 0) >= threshold:
                        filtered.append(msgs[i])
                filtered.extend(msgs[-2:])  # Last N

                total_tokens_after = sum(
                    len(pruner.tokenizer.encode(m["content"]).ids) for m in filtered
                )
                kept_indices = [
                    i
                    for i, m in enumerate(msgs)
                    if any(m["content"] == f["content"] for f in filtered)
                ]
                recall = len([idx for idx in critical if idx in kept_indices]) / len(critical) * 100
                savings = (1 - total_tokens_after / total_tokens_before) * 100

                status = "✅ PASS" if recall >= data["target_recall"] else "❌ FAIL"
                results.append(
                    {
                        "Scenario": name,
                        "Algo": algo,
                        "Mult": mult,
                        "Sav": savings,
                        "Rec": recall,
                        "Status": status,
                    }
                )
                print(
                    f"[{status}] {algo:<10} | mult={mult:.2f} | Sav={savings:>5.1f}% | Rec={recall:>3.0f}%"
                )

    # Save to MD
    with open("benchmarks/RESEARCH_SETFIT_FILTER.md", "w") as f:
        f.write("# Research Step 6: SetFit (MiniLM-L12) as Pruner\n\n")
        for name in scenarios.keys():
            f.write(f"### Scenario: {name}\n")
            f.write("| Algo | Multiplier | Savings % | Recall % | Status |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for r in results:
                if r["Scenario"] == name:
                    f.write(
                        f"| {r['Algo']} | {r['Mult']:.2f} | {r['Sav']:.1f}% | {r['Rec']:.0f}% | {r['Status']} |\n"
                    )
            f.write("\n")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    run_setfit_research()
