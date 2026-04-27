import numpy as np
from src.wamp.core.filter import WAMPruner
from benchmarks.needle_test_long import generate_haystack
from benchmarks.multi_doc_qa_long import generate_multi_doc_long
from benchmarks.coherence_test_long import generate_coherence_long


def run_mass_calibration():
    print("=== MASS LONG-CONTEXT CALIBRATION (No LLM) ===")
    pruner = WAMPruner()

    scenarios = {
        "Needle In Haystack (112 msgs)": {
            "msgs": generate_haystack(needle_idx=25, total_msgs=55),
            "task": "I am writing the user registration endpoint. Which hashing algorithm should I use for the passwords? Answer with just the algorithm name.",
            "critical": [51, 52],  # Argon2id rules indices
        },
        "Multi-Doc QA (114 msgs)": {
            "msgs": generate_multi_doc_long(),
            "task": "Compare the budgets and lead scientists of projects Alpha and Beta. Who has more funding?",
            "critical": [1, 15, 29, 43, 57, 71, 113],  # Indices with Alpha/Beta/Budget info
        },
        "Coherence/Summarization (114 msgs)": {
            "msgs": generate_coherence_long(),
            "task": "Provide a comprehensive summary of the architecture. Mention the database, cache, and monitoring tools explicitly.",
            "critical": [1, 15, 29, 43, 57, 71, 85, 99],  # All spec parts
        },
    }

    multipliers = np.arange(0.90, 1.01, 0.01)

    for name, data in scenarios.items():
        print(f"\n>>> Scenario: {name}")
        msgs = data["msgs"]
        task = f"Analyze the following conversation. Identify and keep all messages that are relevant to answering this specific request: '{data['task']}'"
        critical = data["critical"]

        _, scores, _, baseline = pruner.get_attention_filtered(msgs, task, keep_last_n=2)

        for mult in multipliers:
            threshold = baseline * mult
            kept_indices = [
                i for i, s in enumerate(scores) if i == 0 or i >= len(msgs) - 2 or s >= threshold
            ]

            saved_crit = [idx for idx in critical if idx in kept_indices]
            recall = len(saved_crit) / len(critical) * 100
            savings = (1 - len(kept_indices) / len(msgs)) * 100

            status = "✅ FULL RECALL" if recall == 100 else f"⚠️ {recall:.0f}% RECALL"
            if recall == 0:
                status = "❌ FAILED"

            print(
                f"  Mult {mult:.2f}: Savings {savings:>4.1f}% | {status} | Kept {len(kept_indices)}/{len(msgs)}"
            )


if __name__ == "__main__":
    import logging

    logging.getLogger("src.wamp.core.filter").setLevel(logging.ERROR)
    run_mass_calibration()
