import numpy as np
from src.wamp.core.filter import WAMPruner
from src.wamp.core.config import FILTER_MAX_TOKENS
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_modernbert_research():
    print("=== RESEARCH STEP 7: ModernBERT-BASE AS CONTEXT PRUNER ===")

    # Use the optimized ModernBERT ONNX model
    pruner = WAMPruner(model_dir="./model_modernbert_onnx")

    scenarios = {
        "Needle (Fact Retrieval)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "What was the database port mentioned earlier?",
            "critical": [51, 52],
            "target_recall": 100,
        },
        "Reasoning (Multi-Doc QA)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113],
            "target_recall": 100,
        },
        "Summary (Coherence)": {
            "msgs": generate_coherence_long(),
            "task": "Summarize the technical specification of the architecture.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99],
            "target_recall": 75,
        },
    }

    algorithms = ["mean_max", "cls_max", "mean_mean", "max_max"]
    # Deep search range with 0.01 step to find 100% recall for Reasoning
    multipliers = np.arange(0.70, 1.01, 0.01)

    results = []

    for name, data in scenarios.items():
        print(f"\n--- Testing Scenario: {name} ---")
        msgs = data["msgs"]
        task_query = data["task"]
        critical = data["critical"]

        # Calculate original tokens (approx)
        total_tokens_before = sum(len(pruner.tokenizer.encode(m["content"]).ids) for m in msgs)

        task_prompt = f"Analyze relevance to: '{task_query}'"
        task_enc = pruner.tokenizer.encode(task_prompt)
        task_ids = task_enc.ids

        # Limit task length for window space
        max_task_safe = int(FILTER_MAX_TOKENS * 0.2)
        if len(task_ids) > max_task_safe:
            task_ids = task_ids[:max_task_safe]
        task_len = len(task_ids)

        for algo in algorithms:
            # 1. Calculate scores for all messages (Algo dependent)
            scores_dict = {}
            for i, msg in enumerate(msgs):
                if i == 0 or i >= len(msgs) - 2:
                    continue

                content = pruner.get_content(msg)
                msg_enc = pruner.tokenizer.encode(content)
                msg_ids = msg_enc.ids
                msg_len = len(msg_ids)

                # Sliding Window logic (simplified for research but matches filter.py)
                window_size = FILTER_MAX_TOKENS - task_len
                if window_size <= 0:
                    window_size = 1

                chunk_scores = []
                for start_idx in range(0, max(1, msg_len), window_size):
                    end_idx = min(start_idx + window_size, msg_len)
                    current_msg_chunk = msg_ids[start_idx:end_idx]
                    if not current_msg_chunk:
                        break

                    input_ids = (task_ids + current_msg_chunk)[:FILTER_MAX_TOKENS]
                    ids_np = np.array([input_ids], dtype=np.int64)
                    mask_np = np.ones_like(ids_np)

                    outputs = pruner.session.run(
                        None, {"input_ids": ids_np, "attention_mask": mask_np}
                    )

                    importance = 0
                    chunk_msg_len = len(current_msg_chunk)
                    for layer_name in pruner.pruning_layers:
                        layer_idx = [o.name for o in pruner.session.get_outputs()].index(layer_name)
                        slc = outputs[layer_idx][
                            0, :, :task_len, task_len : task_len + chunk_msg_len
                        ]
                        if slc.size == 0:
                            continue

                        if algo == "mean_max":
                            importance += float(np.mean(np.max(slc, axis=2)))
                        elif algo == "cls_max":
                            importance += float(np.max(slc[:, 0, :]))
                        elif algo == "mean_mean":
                            importance += float(np.mean(slc))
                        elif algo == "max_max":
                            importance += float(np.max(slc))
                        else:
                            importance += float(np.max(slc[:, 0, :]))

                    chunk_scores.append(importance / len(pruner.pruning_layers))
                    if end_idx >= msg_len:
                        break

                scores_dict[i] = max(chunk_scores) if chunk_scores else 0

            # 2. Iterate through multipliers for this algo/scenario
            for mult in multipliers:
                scores = list(scores_dict.values())
                baseline = np.median(scores) if scores else 0
                threshold = baseline * mult

                # Determine which messages to keep
                keep_indices = {0}
                for i in range(max(0, len(msgs) - 2), len(msgs)):
                    keep_indices.add(i)
                for i, msg in enumerate(msgs):
                    if i in keep_indices:
                        continue
                    if scores_dict.get(i, 0) >= threshold:
                        keep_indices.add(i)

                filtered = [msgs[i] for i in sorted(list(keep_indices))]

                total_tokens_after = sum(
                    len(pruner.tokenizer.encode(m["content"]).ids) for m in filtered
                )
                kept_indices = [
                    i
                    for i, m in enumerate(msgs)
                    if any(m["content"] == f["content"] for f in filtered)
                ]
                recall = (
                    len([idx for idx in data["critical"] if idx in kept_indices])
                    / len(data["critical"])
                    * 100
                )
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
    report_path = "benchmarks/RESEARCH_MODERNBERT_FILTER.md"
    with open(report_path, "w") as f:
        f.write("# Research Step 7: ModernBERT-base as Pruner\n\n")
        f.write("*Testing with 2048 token window and sliding window support.*\n\n")
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
    print(f"\n🚀 Research complete! Report saved to {report_path}")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.ERROR)
    run_modernbert_research()
