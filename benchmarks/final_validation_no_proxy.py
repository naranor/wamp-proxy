from src.wamp.core.filter import WAMPruner
from src.wamp.core.config import FILTER_NEEDLE_ALGO, FILTER_REASONING_ALGO, FILTER_SUMMARY_ALGO
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_large_validation():
    print("=== FINAL VALIDATION: UNIFIED SETFIT ENGINE (MINILM-L12) ===")
    # Use the new SetFit model directory
    pruner = WAMPruner(model_dir="./model_setfit_onnx")

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

        # 1. Route and Filter using the updated API
        category, _ = pruner.classify_task(task_query)
        filtered = pruner.get_attention_filtered(msgs, task_query)

        # Get algorithm name for display
        if category == "Reasoning":
            algo = FILTER_REASONING_ALGO
        elif category == "Needle":
            algo = FILTER_NEEDLE_ALGO
        else:
            algo = FILTER_SUMMARY_ALGO

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


if __name__ == "__main__":
    import logging

    logging.getLogger("wamp").setLevel(logging.WARNING)
    run_large_validation()
