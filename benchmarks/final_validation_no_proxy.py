from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_large_validation():
    print("=== FINAL VALIDATION: TRI-MODAL ADAPTIVE ENGINE (V4) ===")
    pruner = WAMPruner(model_dir="./deberta_nli_model")

    scenarios = {
        "Needle (Fact Retrieval)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "What was the database port mentioned earlier?",
            "critical": [51, 52],
        },
        "Reasoning (Multi-Doc QA)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113],
        },
        "Coherence (Architecture Summary)": {
            "msgs": generate_coherence_long(),
            "task": "Summarize the technical specification of the architecture.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99],
        },
    }

    print(f"{'Scenario':<35} | {'Cat':<10} | {'Algo':<10} | {'Sav%':<7} | {'Recall%'}")
    print("-" * 85)

    for name, data in scenarios.items():
        msgs = data["msgs"]
        task_query = data["task"]
        critical = data["critical"]

        task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{task_query}'"

        category = pruner.classify_task(task_query)
        filtered, _, _, _ = pruner.get_attention_filtered(
            msgs, task, original_query=task_query, keep_last_n=2
        )

        if category == "Reasoning":
            algo = "cls_max"
        elif category == "Needle":
            algo = "mean_max"
        else:
            algo = "mean_mean"

        kept_indices = [
            i for i, m in enumerate(msgs) if any(m["content"] == f["content"] for f in filtered)
        ]
        recall = len([idx for idx in critical if idx in kept_indices]) / len(critical) * 100
        savings = (1 - len(filtered) / len(msgs)) * 100

        print(f"{name:<35} | {category:<10} | {algo:<10} | {savings:>6.1f}% | {recall:>7.0f}%")


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.WARNING)
    run_large_validation()
